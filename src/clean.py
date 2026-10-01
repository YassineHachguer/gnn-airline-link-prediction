"""
Nettoyage du graphe des aéroports.

Entrée:
    data/raw/airports.graphml.xml

Sorties:
    data/clean/airports_clean.graphml
    data/clean/node_mapping.csv
    data/clean/nodes_metadata.csv
    data/clean/suspect_nodes.csv

Principe:
- ne pas corriger manuellement les coordonnées/populations ;
- normaliser les pays ;
- conserver uniquement la composante connexe principale ;
- réindexer les nœuds de 0 à N-1 ;
- signaler les populations par défaut et les coordonnées suspectes.
"""

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/raw/airports.graphml.xml")
DEFAULT_OUTPUT = Path("data/clean")


def first_value(value, default=None):
    """Convertit proprement les attributs GraphML parfois stockés en liste."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def normalize_text(value: object) -> str:
    """Normalisation légère pour comparer les noms de pays."""
    text = html.unescape(str(value or "")).strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.upper()


def normalize_country(value: object) -> str:
    """
    Certains pays apparaissent sous la forme:
        M&Eacute;XICO#MEXICO
        VIÊT_NAM#VIET_NAM,VIETNAM

    On garde la première représentation, puis on applique quelques
    normalisations stables.
    """
    raw = html.unescape(str(value or "")).strip()
    candidate = raw.split("#")[0].split(",")[0]
    country = normalize_text(candidate)

    aliases = {
        "VIET NAM": "VIETNAM",
        "VIETNAM": "VIETNAM",
        "UNITED STATES": "UNITED STATES",
        "USA": "UNITED STATES",
        "US": "UNITED STATES",
        "UK": "UNITED KINGDOM",
        "RUSSIA": "RUSSIA",
        "RUSSIAN FEDERATION": "RUSSIA",
    }
    return aliases.get(country, country or "UNKNOWN")


def to_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance orthodromique en kilomètres."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def suspicious_score(G: nx.Graph) -> pd.DataFrame:
    """
    Score simple d'incohérence géographique.

    Pour un nœud v:
      - distance médiane v -> voisins
      - distance médiane entre chaque voisin et ses propres voisins
      - score = ratio des deux.

    Un score élevé signifie que v est beaucoup plus loin de ses voisins
    que ceux-ci ne le sont de leur environnement local.

    Ce score sert à SIGNALER, pas à corriger.
    """
    rows = []

    for node in G.nodes:
        lat = to_float(G.nodes[node].get("lat"))
        lon = to_float(G.nodes[node].get("lon"))
        neighbors = list(G.neighbors(node))

        if not np.isfinite(lat) or not np.isfinite(lon) or len(neighbors) < 2:
            rows.append(
                {
                    "node_id": node,
                    "suspicion_score": np.nan,
                    "n_neighbors": len(neighbors),
                }
            )
            continue

        d_to_neighbors = []
        local_neighbor_distances = []

        for nb in neighbors:
            nlat = to_float(G.nodes[nb].get("lat"))
            nlon = to_float(G.nodes[nb].get("lon"))
            if np.isfinite(nlat) and np.isfinite(nlon):
                d_to_neighbors.append(haversine_km(lat, lon, nlat, nlon))

                for nb2 in G.neighbors(nb):
                    if nb2 == node:
                        continue
                    lat2 = to_float(G.nodes[nb2].get("lat"))
                    lon2 = to_float(G.nodes[nb2].get("lon"))
                    if np.isfinite(lat2) and np.isfinite(lon2):
                        local_neighbor_distances.append(
                            haversine_km(nlat, nlon, lat2, lon2)
                        )

        if not d_to_neighbors or not local_neighbor_distances:
            score = np.nan
        else:
            denominator = np.median(local_neighbor_distances)
            score = float(np.median(d_to_neighbors) / max(denominator, 1e-6))

        rows.append(
            {
                "node_id": node,
                "suspicion_score": score,
                "n_neighbors": len(neighbors),
            }
        )

    return pd.DataFrame(rows)


def clean_graph(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Lecture : {input_path}")
    G = nx.read_graphml(input_path)

    if G.is_directed():
        G = nx.Graph(G)

    # GraphML peut contenir des doublons ou boucles selon la source.
    G.remove_edges_from(nx.selfloop_edges(G))

    # Nettoyage des attributs sans modification des données numériques.
    for node in G.nodes:
        attrs = G.nodes[node]
        attrs["lat"] = to_float(attrs.get("lat"))
        attrs["lon"] = to_float(attrs.get("lon"))
        attrs["population"] = to_float(attrs.get("population"))
        attrs["country_raw"] = str(attrs.get("country", ""))
        attrs["country"] = normalize_country(attrs.get("country"))
        attrs["city_name"] = str(attrs.get("city_name", ""))

        # 10 000 est conservé mais marqué comme valeur par défaut.
        attrs["population_default"] = bool(
            np.isfinite(attrs["population"]) and attrs["population"] == 10000
        )

    print(f"    {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    # Conserver la composante connexe principale.
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    G = G.subgraph(largest).copy()

    print(
        f"[2/6] Composante principale : "
        f"{G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes"
    )

    # Score de suspicion AVANT réindexation pour conserver l'ID original.
    print("[3/6] Calcul du score de suspicion...")
    suspicion = suspicious_score(G)

    # Réindexation exigée par PyTorch Geometric.
    old_nodes = list(G.nodes)
    mapping = {old_id: new_id for new_id, old_id in enumerate(old_nodes)}
    G = nx.relabel_nodes(G, mapping, copy=True)

    suspicion["old_node_id"] = suspicion["node_id"].astype(str)
    suspicion["node_id"] = suspicion["node_id"].map(mapping)
    suspicion = suspicion.dropna(subset=["node_id"])
    suspicion["node_id"] = suspicion["node_id"].astype(int)

    # Métadonnées de chaque nœud.
    metadata = []
    for node in G.nodes:
        attrs = G.nodes[node]
        metadata.append(
            {
                "node_id": int(node),
                "city_name": attrs.get("city_name", ""),
                "country": attrs.get("country", "UNKNOWN"),
                "country_raw": attrs.get("country_raw", ""),
                "lat": attrs.get("lat", np.nan),
                "lon": attrs.get("lon", np.nan),
                "population": attrs.get("population", np.nan),
                "population_default": int(attrs.get("population_default", False)),
                "degree_full": G.degree(node),
            }
        )

    metadata = pd.DataFrame(metadata)

    # Les 30 plus suspects sont seulement signalés.
    top30 = (
        suspicion.dropna(subset=["suspicion_score"])
        .sort_values("suspicion_score", ascending=False)
        .head(30)
        .copy()
    )
    top30["suspect_rank"] = np.arange(1, len(top30) + 1)
    suspect_ids = set(top30["node_id"].astype(int))

    nx.set_node_attributes(
        G,
        {n: int(n in suspect_ids) for n in G.nodes},
        "suspect",
    )
    score_map = suspicion.set_index("node_id")["suspicion_score"].to_dict()
    nx.set_node_attributes(G, score_map, "suspicion_score")

    print("[4/6] Écriture des fichiers...")

    graph_path = output_dir / "airports_clean.graphml"
    nx.write_graphml(G, graph_path)

    mapping_df = pd.DataFrame(
        {
            "new_id": list(range(len(old_nodes))),
            "original_id": old_nodes,
        }
    )
    mapping_df.to_csv(output_dir / "node_mapping.csv", index=False)

    metadata = metadata.merge(
        suspicion[["node_id", "suspicion_score"]],
        on="node_id",
        how="left",
    )
    metadata["suspect"] = metadata["node_id"].isin(suspect_ids).astype(int)
    metadata.to_csv(output_dir / "nodes_metadata.csv", index=False)

    top30.to_csv(output_dir / "suspect_nodes.csv", index=False)

    print("[5/6] Vérifications...")
    assert min(G.nodes) == 0
    assert max(G.nodes) == G.number_of_nodes() - 1
    assert not any(u == v for u, v in G.edges)

    print("[6/6] Terminé.")
    print(f"    Graphe       : {graph_path}")
    print(f"    Mapping      : {output_dir / 'node_mapping.csv'}")
    print(f"    Métadonnées  : {output_dir / 'nodes_metadata.csv'}")
    print(f"    Suspects     : {output_dir / 'suspect_nodes.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    clean_graph(args.input, args.output)


if __name__ == "__main__":
    main()
