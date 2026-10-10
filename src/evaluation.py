import itertools
import numpy as np
import pandas as pd
import torch as th
from torch_geometric.transforms import RandomLinkSplit

from .data import make_data
from .features import build_features, build_coords
from .baselines import adamic_adar, run_mlp
from .gnn_models import build_model
from .training import train_model, test_model

AA_NAME = "Adamic-Adar (structure seule)"
MLP_NAME = "MLP (features seules, sans graphe)"

CONVS = ["gcn", "sage", "gat"]
DECS = ["dot", "mlp", "mlp_dist"]
DEC_LABEL = {"dot": "produit scalaire", "mlp": "MLP", "mlp_dist": "MLP + distance"}

DEFAULT_HP = dict(hidden=64, out=32, lr=0.01)


def _grid():
    g = {}
    for conv, dec in itertools.product(CONVS, DECS):
        name = f"VGAE {conv.upper()} + {DEC_LABEL[dec]}"
        g[name] = dict(var=True, layers=2, coord=True, pop=True, ctry=True, conv=conv, dec=dec)
    return g

GRID = _grid()
GRID_NAMES = {(c["conv"], c["dec"]): n for n, c in GRID.items()}   
BASE_CFG = GRID["VGAE GCN + produit scalaire"]                      


def ablation_configs(best):
    abl = {
        "  - sans coordonnees (encodeur)": {**best, "coord": False},
        "  - sans population":             {**best, "pop": False},
        "  - sans pays":                   {**best, "ctry": False},
        "  - sans aucune feature (encodeur)": {**best, "coord": False, "pop": False, "ctry": False},
        "  - GAE (sans partie variationnelle)": {**best, "var": False},
        "  - 1 couche au lieu de 2":       {**best, "layers": 1},
    }
    if best["conv"] != "gcn":
        abl["  - encodeur GCN"] = {**best, "conv": "gcn"}
    if best["dec"] != "dot":
        abl["  - decodeur produit scalaire"] = {**best, "dec": "dot"}
    if best["dec"] == "mlp_dist":
        abl["  - decodeur MLP sans distance"] = {**best, "dec": "mlp"}
    return abl


def split_data(data, test_ratio, seed, val_ratio=0.05):
    th.manual_seed(seed)
    split = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True,
                            split_labels=True, add_negative_train_samples=False)
    return split(data)   


def set_features(splits, x):
    for d in splits:
        d.x = x
    return splits


def _model_from_cfg(cfg, x, hp, coords):
    return build_model(x.shape[1], hp["hidden"], hp["out"], cfg["layers"], cfg["var"],
                       cfg["conv"], cfg["dec"], coords)


def run_gnn(cfg, splits, x, seed, hp=DEFAULT_HP, epochs=200, coords=None):
    set_features(splits, x)
    train_d, val_d, test_d = splits
    th.manual_seed(seed)
    model = _model_from_cfg(cfg, x, hp, coords)
    out = train_model(model, train_d, val_d, epochs, cfg["var"], hp["lr"])
    auc, ap = test_model(model, test_d)
    return auc, ap, out["best_val_auc"]


def select_hyperparams(edge_index, info, grid, ratio=0.10, seeds=(0, 1), epochs=200):
    x = build_features(info)
    coords = build_coords(info)
    rows = []
    for hidden, out, lr in itertools.product(grid["hidden"], grid["out"], grid["lr"]):
        hp = dict(hidden=hidden, out=out, lr=lr)
        vals = []
        for seed in seeds:
            splits = split_data(make_data(x, edge_index), ratio, seed)
            vals.append(run_gnn(BASE_CFG, splits, x, seed, hp, epochs, coords)[2])
        rows.append(dict(hidden=hidden, out=out, lr=lr, val_auc=np.mean(vals)))
    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    return dict(hidden=int(best.hidden), out=int(best.out), lr=float(best.lr)), df


def run_all(edge_index, info, ratios=(0.10, 0.20), seeds=(0, 1, 2, 3, 4),
            hp=DEFAULT_HP, epochs=200, verbose=True):
    x_full = build_features(info)
    coords = build_coords(info)
    x_cache = {}

    def get_x(cfg):
        key = (cfg["coord"], cfg["pop"], cfg["ctry"])
        if key not in x_cache:
            x_cache[key] = build_features(info, *key)
        return x_cache[key]

    def get_splits(ratio, seed):
        return set_features(split_data(make_data(x_full, edge_index), ratio, seed), x_full)

    results, val_scores = {}, {}

    # ---- etape 1 : baselines + grille ----
    for ratio in ratios:
        for seed in seeds:
            splits = get_splits(ratio, seed)
            results.setdefault((AA_NAME, ratio), []).append(adamic_adar(splits))
            results.setdefault((MLP_NAME, ratio), []).append(
                run_mlp(splits, seed, epochs, hp["hidden"], hp["out"], hp["lr"]))
            for name, cfg in GRID.items():
                auc, ap, val = run_gnn(cfg, splits, get_x(cfg), seed, hp, epochs, coords)
                results.setdefault((name, ratio), []).append((auc, ap))
                val_scores.setdefault(name, []).append(val)
        if verbose:
            print(f"[etape 1] ratio {ratio} termine")

    best_name = max(val_scores, key=lambda n: np.mean(val_scores[n]))
    best_cfg = GRID[best_name]
    abl = ablation_configs(best_cfg)
    if verbose:
        print("meilleur modele (AUC validation) :", best_name)

    # ---- etape 2 : ablation du meilleur modele ----
    for ratio in ratios:
        for seed in seeds:
            splits = get_splits(ratio, seed)
            for name, cfg in abl.items():
                auc, ap, _ = run_gnn(cfg, splits, get_x(cfg), seed, hp, epochs, coords)
                results.setdefault((name, ratio), []).append((auc, ap))
        if verbose:
            print(f"[etape 2] ratio {ratio} termine")

    meta = dict(best_name=best_name, best_cfg=best_cfg,
                rows=[AA_NAME, MLP_NAME] + list(GRID) + list(abl),
                val_auc={n: float(np.mean(v)) for n, v in val_scores.items()})
    return results, meta


def make_table(results, ratios, rows):
    cols = [(f"{int(r*100)}% caches", m) for r in ratios for m in ("AUC", "AP")]
    mean = pd.DataFrame(index=rows, columns=pd.MultiIndex.from_tuples(cols), dtype=float)
    std = mean.copy()
    for name in rows:
        for r in ratios:
            arr = np.array(results[(name, r)])
            for j, m in enumerate(("AUC", "AP")):
                mean.loc[name, (f"{int(r*100)}% caches", m)] = arr[:, j].mean()
                std.loc[name, (f"{int(r*100)}% caches", m)] = arr[:, j].std()
    table = mean.copy().astype(object)
    for c in mean.columns:
        best = mean[c].idxmax()
        for name in rows:
            s = f"{mean.loc[name, c]:.3f} ± {std.loc[name, c]:.3f}"
            table.loc[name, c] = f"**{s}**" if name == best else s
    return table, mean, std


def fit_final(cfg, edge_index, info, hp=DEFAULT_HP, ratio=0.10, seed=0, epochs=200):
    x = build_features(info, cfg["coord"], cfg["pop"], cfg["ctry"])
    coords = build_coords(info)
    splits = split_data(make_data(x, edge_index), ratio, seed)
    th.manual_seed(seed)
    model = _model_from_cfg(cfg, x, hp, coords)
    out = train_model(model, splits[0], splits[1], epochs, cfg["var"], hp["lr"])
    return model, splits, out


def suspicious_links(model, train_data, info, k=15):
    model.eval()
    with th.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        ei = train_data.pos_edge_label_index
        scores = model.decode(z, ei)          
    out = []
    for i in scores.argsort()[:k]:
        u, v = ei[0, i].item(), ei[1, i].item()
        out.append((info["names"][u], info["country"][u], info["names"][v],
                    info["country"][v], scores[i].item()))
    return pd.DataFrame(out, columns=["ville_1", "pays_1", "ville_2", "pays_2", "score"])
