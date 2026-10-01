"""
Séparation des arêtes en train / validation / test.

Pour chaque graine (0 à 4) et chaque scénario (10 % et 20 % de test) :
- validation : 5 % des arêtes ; test : 10 % ou 20 % ;
- les arêtes d'un arbre couvrant restent toujours en train, donc aucun
  aéroport n'est isolé et le graphe d'entraînement reste connexe ;
- deux jeux de négatifs pour la validation et le test :
    random : paires non reliées tirées au hasard ;
    hard   : pour chaque route (u, v), une paire (u, w) avec w parmi les
             20 aéroports les plus proches de u non reliés à u.

Les aéroports mal localisés sont conservés (bruit gardé volontairement).
Les négatifs d'entraînement sont tirés pendant l'entraînement, pas ici.

Sortie : data/splits/test{10,20}_seed{0..4}.npz

Exécution :
    python src/split.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


DEFAULT_CLEAN = Path("data/clean")
DEFAULT_OUTPUT = Path("data/splits")

SEEDS = [0, 1, 2, 3, 4]
TEST_RATIOS = [0.10, 0.20]
VAL_RATIO = 0.05
K_HARD = 20   # w est tiré parmi les 20 plus proches aéroports NON reliés à u
K_MAX = 300   # voisins géographiques précalculés (marge pour les hubs, dont les plus proches sont souvent déjà reliés)


def load_clean(clean_dir: Path):
    meta = pd.read_csv(clean_dir / "nodes_metadata.csv").sort_values("node_id").reset_index(drop=True)
    G = nx.read_graphml(clean_dir / "airports_clean.graphml")
    edges = np.array(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in G.edges))
    return meta, edges


def random_spanning_tree(n_nodes: int, edges: np.ndarray, rng) -> np.ndarray:
    """Masque booléen des arêtes d'un arbre couvrant aléatoire (Kruskal)."""
    parent = np.arange(n_nodes)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    in_tree = np.zeros(len(edges), dtype=bool)
    for i in rng.permutation(len(edges)):
        ru, rv = find(edges[i, 0]), find(edges[i, 1])
        if ru != rv:
            parent[ru] = rv
            in_tree[i] = True
    return in_tree


def geo_knn(meta: pd.DataFrame, k: int) -> np.ndarray:
    """Indices des k aéroports les plus proches de chaque aéroport (hors lui-même)."""
    lat = np.radians(meta["lat"].to_numpy())
    lon = np.radians(meta["lon"].to_numpy())
    # Coordonnées 3D sur la sphère : la distance euclidienne donne le même classement que la distance haversine
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    n = len(xyz)
    knn = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, 500):
        block = xyz[start:start + 500]
        d = ((block[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
        d[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf  # exclut le nœud lui-même
        idx = np.argpartition(d, k, axis=1)[:, :k]
        order = np.take_along_axis(d, idx, axis=1).argsort(axis=1)
        knn[start:start + 500] = np.take_along_axis(idx, order, axis=1)
    return knn


def sample_random_negatives(n_nodes, n_samples, edge_set, rng, exclude=set()):
    neg = set()
    while len(neg) < n_samples:
        u, v = rng.integers(0, n_nodes, size=2)
        pair = (min(u, v), max(u, v))
        if u != v and pair not in edge_set and pair not in exclude:
            neg.add(pair)
    return np.array(sorted(neg), dtype=np.int64)


def sample_hard_negatives(pos, knn, edge_set, rng, exclude=set()):
    """Pour chaque route (u, v) : (u, w) avec w parmi les K_HARD plus proches de u non reliés à u."""
    neg = set()
    for u, v in pos:
        # On alterne l'extrémité de référence ; si elle n'a aucun candidat, on essaie l'autre
        anchors = (u, v) if rng.random() < 0.5 else (v, u)
        for anchor in anchors:
            candidates = []
            for w in knn[anchor]:
                pair = (min(anchor, w), max(anchor, w))
                if pair not in edge_set and pair not in neg and pair not in exclude:
                    candidates.append(pair)
                    if len(candidates) == K_HARD:
                        break
            if candidates:
                neg.add(candidates[rng.integers(len(candidates))])
                break
    return np.array(sorted(neg), dtype=np.int64)


def make_split(meta, edges, test_ratio, seed, knn):
    rng = np.random.default_rng(seed)
    n_nodes, n_edges = len(meta), len(edges)
    edge_set = set(map(tuple, edges))

    # Les arêtes de l'arbre couvrant restent en train ; val et test sont tirés parmi les autres
    in_tree = random_spanning_tree(n_nodes, edges, rng)
    candidates = rng.permutation(np.flatnonzero(~in_tree))

    n_test = int(round(test_ratio * n_edges))
    n_val = int(round(VAL_RATIO * n_edges))
    assert n_test + n_val <= len(candidates), "pas assez d'arêtes hors arbre couvrant"

    test_idx = candidates[:n_test]
    val_idx = candidates[n_test:n_test + n_val]
    train_mask = np.ones(n_edges, dtype=bool)
    train_mask[test_idx] = False
    train_mask[val_idx] = False

    split = {
        "train_pos": edges[train_mask],
        "val_pos": edges[val_idx],
        "test_pos": edges[test_idx],
    }

    # Négatifs : jamais une vraie route (edge_set contient toutes les arêtes), jamais en double entre val et test
    split["val_neg_random"] = sample_random_negatives(n_nodes, n_val, edge_set, rng)
    split["test_neg_random"] = sample_random_negatives(
        n_nodes, n_test, edge_set, rng, exclude=set(map(tuple, split["val_neg_random"])))

    split["val_neg_hard"] = sample_hard_negatives(split["val_pos"], knn, edge_set, rng)
    split["test_neg_hard"] = sample_hard_negatives(
        split["test_pos"], knn, edge_set, rng, exclude=set(map(tuple, split["val_neg_hard"])))

    check_split(split, edge_set, n_nodes)
    return split


def check_split(split, edge_set, n_nodes):
    pos_sets = [set(map(tuple, split[k])) for k in ("train_pos", "val_pos", "test_pos")]
    assert not (pos_sets[0] & pos_sets[1]) and not (pos_sets[0] & pos_sets[2]) and not (pos_sets[1] & pos_sets[2])
    assert sum(len(s) for s in pos_sets) == len(edge_set)

    # Autant de négatifs que de positifs (AUC et AP comparables entre méthodes)
    assert len(split["val_neg_hard"]) == len(split["val_pos"])
    assert len(split["test_neg_hard"]) == len(split["test_pos"])

    for key in ("val_neg_random", "test_neg_random", "val_neg_hard", "test_neg_hard"):
        neg = split[key]
        assert not (set(map(tuple, neg)) & edge_set), f"{key} contient une vraie route"
        assert (neg[:, 0] != neg[:, 1]).all()

    # Graphe d'entraînement connexe : aucun aéroport isolé
    G_train = nx.Graph()
    G_train.add_nodes_from(range(n_nodes))
    G_train.add_edges_from(map(tuple, split["train_pos"]))
    assert nx.is_connected(G_train)


def load_split(path):
    """Charge un split sauvegardé : dictionnaire de tableaux (P, 2)."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    meta, edges = load_clean(args.clean)
    knn = geo_knn(meta, K_MAX)
    print(f"{len(meta)} nœuds, {len(edges)} arêtes\n")

    summary = []
    for test_ratio in TEST_RATIOS:
        for seed in SEEDS:
            split = make_split(meta, edges, test_ratio, seed, knn)
            name = f"test{int(test_ratio * 100)}_seed{seed}"
            np.savez(args.output / f"{name}.npz", **split)

            sizes = {k: len(v) for k, v in split.items()}
            summary.append({"split": name, **sizes})
            print(f"{name} : " + ", ".join(f"{k}={v}" for k, v in sizes.items()))

    config = {"seeds": SEEDS, "test_ratios": TEST_RATIOS, "val_ratio": VAL_RATIO, "k_hard": K_HARD, "k_max": K_MAX,
              "splits": summary}
    (args.output / "splits_info.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n{len(summary)} splits écrits dans {args.output}")


if __name__ == "__main__":
    main()
