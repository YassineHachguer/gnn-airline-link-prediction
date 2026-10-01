"""
Construction des features pour la prédiction de liens.

- Features des nœuds (x) : sin/cos des coordonnées, log(population),
  indicateur population manquante, log(degré).
- Features des paires (décodeur) : distance, même pays.

Le degré est calculé uniquement sur les arêtes d'entraînement (anti-fuite).

Test rapide :
    python src/features.py --metadata data/clean/nodes_metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METADATA = Path("data/clean/nodes_metadata.csv")
EARTH_RADIUS_KM = 6371.0088

# Chaque feature peut être désactivée pour l'ablation.
DEFAULT_CONFIG = {
    "node": {
        "coords_sincos": True,
        "log_pop": True,
        "pop_missing": True,
        "degree": True,
    },
    "pair": {
        "distance": True,
        "same_country": True,
    },
}


def load_metadata(path=DEFAULT_METADATA) -> pd.DataFrame:
    meta = pd.read_csv(path).sort_values("node_id").reset_index(drop=True)
    # La ligne i de x doit correspondre au nœud i
    assert (meta["node_id"].to_numpy() == np.arange(len(meta))).all()
    return meta


def standardize(values: np.ndarray) -> np.ndarray:
    std = values.std()
    return (values - values.mean()) / (std if std > 0 else 1.0)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def degree_from_edges(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    # edges : (E, 2), chaque arête une seule fois (pas dans les deux sens)
    edges = np.asarray(edges, dtype=np.int64)
    deg = np.bincount(edges[:, 0], minlength=n_nodes) + np.bincount(edges[:, 1], minlength=n_nodes)
    return deg.astype(np.float64)


def build_node_features(meta, train_edges, config=None):
    """Retourne x (N, F) et la liste des noms de colonnes."""
    cfg = (config or DEFAULT_CONFIG)["node"]
    n = len(meta)
    columns, names = [], []

    if cfg.get("coords_sincos"):
        # sin/cos car la longitude est cyclique (-180° = 180°)
        lat = np.radians(meta["lat"].to_numpy())
        lon = np.radians(meta["lon"].to_numpy())
        columns += [np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)]
        names += ["sin_lat", "cos_lat", "sin_lon", "cos_lon"]

    if cfg.get("log_pop"):
        columns.append(standardize(np.log10(meta["population"].to_numpy())))
        names.append("log_pop")

    if cfg.get("pop_missing"):
        columns.append(meta["population_default"].to_numpy().astype(np.float64))
        names.append("pop_missing")

    if cfg.get("degree"):
        # Uniquement les arêtes d'entraînement, sinon fuite de données
        deg = degree_from_edges(n, train_edges)
        columns.append(standardize(np.log1p(deg)))
        names.append("log_degree_train")

    if not columns:
        # Aucune feature : le modèle n'utilise que la structure
        columns.append(np.ones(n))
        names.append("constant")

    x = np.stack(columns, axis=1).astype(np.float32)
    assert np.isfinite(x).all()
    return x, names


def build_pair_features(meta, pairs, config=None, distance_stats=None):
    """
    Retourne les features des paires (P, F), leurs noms et les stats de distance.

    distance_stats : à calculer sur l'entraînement (laisser None),
    puis à réutiliser telles quelles pour la validation et le test.
    """
    cfg = (config or DEFAULT_CONFIG)["pair"]
    pairs = np.asarray(pairs, dtype=np.int64)
    u, v = pairs[:, 0], pairs[:, 1]
    columns, names = [], []

    if cfg.get("distance"):
        lat, lon = meta["lat"].to_numpy(), meta["lon"].to_numpy()
        log_d = np.log1p(haversine_km(lat[u], lon[u], lat[v], lon[v]))
        if distance_stats is None:
            distance_stats = (float(log_d.mean()), float(log_d.std() or 1.0))
        mean, std = distance_stats
        columns.append((log_d - mean) / std)
        names.append("log_distance")

    if cfg.get("same_country"):
        country = meta["country"].to_numpy()
        columns.append((country[u] == country[v]).astype(np.float64))
        names.append("same_country")

    if not columns:
        return np.zeros((len(pairs), 0), dtype=np.float32), [], distance_stats

    feats = np.stack(columns, axis=1).astype(np.float32)
    assert np.isfinite(feats).all()
    return feats, names, distance_stats


def sanity_check(metadata_path: Path, graph_path: Path | None):
    """Vérifie que tout tourne. Utilise toutes les arêtes : test uniquement."""
    import networkx as nx

    meta = load_metadata(metadata_path)
    G = nx.read_graphml(graph_path or metadata_path.parent / "airports_clean.graphml")
    edges = np.array([(int(a), int(b)) for a, b in G.edges])  # ids relus en str

    print(f"{len(meta)} nœuds, {len(edges)} arêtes (toutes les arêtes : test uniquement)\n")

    x, node_names = build_node_features(meta, edges)
    print(f"x : {x.shape}")
    print(pd.DataFrame(x, columns=node_names).describe().round(3).T[["mean", "std", "min", "max"]], "\n")

    rng = np.random.default_rng(0)
    rand = rng.integers(0, len(meta), size=(len(edges), 2))
    rand = rand[rand[:, 0] != rand[:, 1]]

    pos, pair_names, stats = build_pair_features(meta, edges)
    neg, _, _ = build_pair_features(meta, rand, distance_stats=stats)
    print(pd.DataFrame({"routes": pos.mean(0), "aléatoires": neg.mean(0)}, index=pair_names).round(3))

    # Ablation : désactiver une feature retire bien sa colonne
    cfg = {"node": {**DEFAULT_CONFIG["node"], "log_pop": False}, "pair": DEFAULT_CONFIG["pair"]}
    x_abl, names_abl = build_node_features(meta, edges, cfg)
    assert "log_pop" not in names_abl and x_abl.shape[1] == x.shape[1] - 1
    print("\nAblation : OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--graph", type=Path, default=None)
    args = parser.parse_args()
    sanity_check(args.metadata, args.graph)


if __name__ == "__main__":
    main()