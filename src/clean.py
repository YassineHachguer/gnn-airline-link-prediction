"""
Nettoyage du graphe des aéroports.

Entrée :
    data/raw/airports.graphml.xml

Sorties :
    data/clean/airports_clean.graphml   graphe nettoyé (nœuds numérotés de 0 à N-1)
    data/clean/node_mapping.csv         correspondance nouvel identifiant -> identifiant d'origine
    data/clean/nodes_metadata.csv       une ligne par aéroport (attributs, indicateurs, score)
    data/clean/suspect_nodes.csv        aéroports les plus suspects, à vérifier manuellement

Principes (issus de l'exploration, notebooks/01_exploration.ipynb) :
- le fichier brut n'est jamais modifié ;
- les coordonnées et les populations ne sont pas corrigées manuellement :
  les problèmes sont signalés, pas corrigés ;
- les noms de pays sont normalisés (Q0) ;
- seule la composante connexe principale est conservée (Q2) ;
- les nœuds sont renumérotés de 0 à N-1, comme l'exige PyTorch Geometric (Q0) ;
- les populations par défaut (10 000) sont signalées par un indicateur (Q5) ;
- un score de suspicion géographique est calculé à des fins d'analyse (Q6).

Exécution :
    python src/clean.py --input data/raw/airports.graphml.xml --output data/clean
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

# Valeur attribuée par défaut aux villes dont la population est inconnue (Q5).
DEFAULT_POPULATION = 10000

# Nombre d'aéroports suspects exportés pour vérification manuelle.
# Il ne s'agit PAS d'un seuil de décision : la liste sert uniquement
# à orienter la vérification à la main (voir suspicious_score).
N_SUSPECTS_TO_REVIEW = 50


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def first_value(value, default=None):
    """Convertit proprement les attributs GraphML parfois stockés en liste."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def normalize_text(value: object) -> str:
    """
    Normalisation légère d'un texte :
    décodage des entités HTML, suppression des accents,
    remplacement des '_' par des espaces, passage en majuscules.
    """
    text = html.unescape(str(value or "")).strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.upper()


def normalize_country(value: object) -> str:
    """
    Normalise le nom d'un pays (Q0).

    Certains pays apparaissent sous plusieurs variantes, par exemple :
        M&Eacute;XICO#MEXICO
        VI&Ecirc;T_NAM#VIET_NAM,VIETNAM

    On conserve la première variante, puis on la normalise
    (accents supprimés, majuscules) et on applique quelques alias.
    """
    raw = html.unescape(str(value or "")).strip()
    candidate = raw.split("#")[0].split(",")[0]
    country = normalize_text(candidate)

    aliases = {
        "VIET NAM": "VIETNAM",
        "USA": "UNITED STATES",
    }
    return aliases.get(country, country or "UNKNOWN")


def to_float(value, default=np.nan):
    """Conversion en nombre réel ; renvoie NaN si la valeur est invalide."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance orthodromique (à la surface de la Terre) en kilomètres."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Score de suspicion géographique (Q6)
# ---------------------------------------------------------------------------

def suspicious_score(G: nx.Graph) -> pd.DataFrame:
    """
    Score d'incohérence géographique d'un aéroport par rapport à ses voisins.

    Idée : les routes du fichier sont fiables, mais certaines coordonnées sont
    erronées (appariement par nom de ville). Un aéroport mal localisé est très
    éloigné de TOUS ses voisins, alors que ces voisins sont proches des leurs.

    Pour un aéroport v :
        med(v)    = distance médiane entre v et ses voisins
        score(v)  = med(v) / (médiane des med(u) pour u voisin de v  + 1)

    - score proche de 1 : v est aussi proche de ses voisins que ceux-ci le sont
      des leurs, situation normale ;
    - score très élevé  : v est loin de ses voisins alors que ceux-ci sont
      cohérents entre eux, ses coordonnées sont probablement fausses.

    Le dénominateur évite d'accuser le voisin innocent d'un aéroport erroné :
    Essaouira (bien placée) a pour unique voisin « Casablanca », placée par
    erreur en Australie ; comme Casablanca est lui-même loin de tous ses
    voisins, le rapport reste faible pour Essaouira.

    Le terme « + 1 » (en km) évite une division par zéro.

    CORRECTIONS par rapport à la version précédente :
    1. Les aéroports n'ayant qu'un seul voisin sont désormais inclus.
       Auparavant, la condition « au moins 2 voisins » excluait 715 aéroports,
       dont de véritables erreurs (Maloelap Island, Green Island, Wasum...).
    2. Le dénominateur est la médiane des médianes de chaque voisin, et non
       la médiane de toutes les distances des voisins de voisins mises en
       commun. L'ancienne version donnait un poids écrasant aux hubs
       (un voisin de 200 routes comptait 200 fois). Le calcul est désormais
       identique à celui du notebook d'exploration.

    Ce score sert à SIGNALER des cas à vérifier manuellement, pas à corriger
    les données, et il ne doit pas être utilisé comme feature du modèle
    (il est calculé sur toutes les arêtes, voir l'avertissement plus bas).
    """

    def dist(a, b):
        return haversine_km(
            G.nodes[a]["lat"], G.nodes[a]["lon"],
            G.nodes[b]["lat"], G.nodes[b]["lon"],
        )

    # Distance médiane de chaque aéroport à ses voisins.
    # Dans la composante principale, chaque nœud a au moins un voisin.
    med = {v: float(np.median([dist(v, u) for u in G.neighbors(v)])) for v in G.nodes}

    rows = []
    for v in G.nodes:
        neighbors_med = np.median([med[u] for u in G.neighbors(v)])
        rows.append(
            {
                "node_id": v,
                "median_km_to_neighbors": med[v],
                "suspicion_score": med[v] / (neighbors_med + 1.0),
                "n_neighbors": G.degree(v),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline de nettoyage
# ---------------------------------------------------------------------------

def clean_graph(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Lecture : {input_path}")
    G = nx.read_graphml(input_path)

    # Contrôles défensifs : le fichier actuel est non orienté et sans boucle
    # (vérifié en Q0), mais ces lignes protègent contre une autre version du fichier.
    if G.is_directed():
        G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Normalisation des attributs, sans modification des valeurs numériques.
    for node in G.nodes:
        attrs = G.nodes[node]
        attrs["lat"] = to_float(attrs.get("lat"))
        attrs["lon"] = to_float(attrs.get("lon"))
        attrs["population"] = to_float(attrs.get("population"))
        attrs["country_raw"] = str(attrs.get("country", ""))       # valeur d'origine conservée
        attrs["country"] = normalize_country(attrs.get("country"))  # valeur normalisée (Q0)
        attrs["city_name"] = str(attrs.get("city_name", ""))

        # La valeur 10 000 est conservée, mais signalée comme valeur par défaut (Q5) :
        # le fait que la population soit manquante est lui-même informatif.
        attrs["population_default"] = bool(
            np.isfinite(attrs["population"]) and attrs["population"] == DEFAULT_POPULATION
        )

    print(f"    {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    # Conservation de la composante connexe principale (Q2) :
    # elle regroupe 99,6 % des nœuds ; les 5 autres composantes (12 nœuds)
    # compliqueraient la séparation des liens sans apporter d'information.
    largest = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest).copy()

    print(
        f"[2/6] Composante principale : "
        f"{G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes"
    )

    # Score de suspicion calculé AVANT la renumérotation,
    # afin de conserver le lien avec l'identifiant d'origine.
    print("[3/6] Calcul du score de suspicion...")
    suspicion = suspicious_score(G)

    # Renumérotation de 0 à N-1, exigée par PyTorch Geometric (Q0).
    # CORRECTION : les nœuds sont triés par identifiant d'origine avant la
    # renumérotation. Sans ce tri, l'ordre dépendait du parcours du graphe
    # (par exemple, le nouvel identifiant 3267 correspondait à l'original 1923).
    old_nodes = sorted(G.nodes, key=int)
    mapping = {old_id: new_id for new_id, old_id in enumerate(old_nodes)}
    G = nx.relabel_nodes(G, mapping, copy=True)

    suspicion["old_node_id"] = suspicion["node_id"].astype(str)
    suspicion["node_id"] = suspicion["node_id"].map(mapping).astype(int)

    # -----------------------------------------------------------------------
    # AVERTISSEMENT : FUITE DE DONNÉES
    # degree_full et suspicion_score sont calculés sur TOUTES les arêtes,
    # y compris celles qui seront masquées dans les ensembles de validation
    # et de test. Ils servent uniquement à l'analyse.
    # Ils ne doivent JAMAIS être utilisés comme features du modèle :
    # le modèle verrait indirectement les routes qu'il doit retrouver.
    # Le degré sera recalculé sur le seul graphe d'entraînement dans features.py.
    # -----------------------------------------------------------------------
    metadata = []
    for node in G.nodes:
        attrs = G.nodes[node]
        metadata.append(
            {
                "node_id": int(node),
                "original_id": old_nodes[node],
                "city_name": attrs.get("city_name", ""),
                "country": attrs.get("country", "UNKNOWN"),
                "country_raw": attrs.get("country_raw", ""),
                "lat": attrs.get("lat", np.nan),
                "lon": attrs.get("lon", np.nan),
                "population": attrs.get("population", np.nan),
                "population_default": int(attrs.get("population_default", False)),
                "degree_full": G.degree(node),  # analyse uniquement (voir avertissement)
            }
        )

    metadata = pd.DataFrame(metadata).sort_values("node_id")
    metadata = metadata.merge(
        suspicion[["node_id", "median_km_to_neighbors", "suspicion_score"]],
        on="node_id",
        how="left",
    )

    # CORRECTION : on n'attribue plus d'étiquette « suspect » à partir d'un seuil
    # arbitraire (anciennement les 30 premiers). Ce seuil laissait de côté des
    # erreurs manifestes, comme Ottawa ou Jersey, localisés en Australie.
    # Le score complet est conservé pour tous les aéroports, et les plus suspects
    # sont exportés pour vérification manuelle à l'aide des noms de leurs voisins.
    names = metadata.set_index("node_id")["city_name"]
    to_review = (
        metadata.sort_values("suspicion_score", ascending=False)
        .head(N_SUSPECTS_TO_REVIEW)
        .copy()
    )
    to_review["suspect_rank"] = np.arange(1, len(to_review) + 1)
    to_review["neighbors"] = [
        ", ".join(sorted(names[u] for u in G.neighbors(v))) for v in to_review["node_id"]
    ]
    # Colonne à remplir manuellement (1 = erreur confirmée, 0 = faux positif).
    to_review["verified_error"] = ""

    # Le score est ajouté comme attribut du graphe, à des fins d'analyse uniquement.
    nx.set_node_attributes(
        G, metadata.set_index("node_id")["suspicion_score"].to_dict(), "suspicion_score"
    )

    print("[4/6] Écriture des fichiers...")

    graph_path = output_dir / "airports_clean.graphml"
    nx.write_graphml(G, graph_path)

    mapping_df = pd.DataFrame(
        {"new_id": list(range(len(old_nodes))), "original_id": old_nodes}
    )
    mapping_df.to_csv(output_dir / "node_mapping.csv", index=False)

    metadata.to_csv(output_dir / "nodes_metadata.csv", index=False)

    to_review[
        ["suspect_rank", "node_id", "original_id", "city_name", "country",
         "lat", "lon", "degree_full",
         "median_km_to_neighbors", "suspicion_score", "neighbors", "verified_error"]
    ].to_csv(output_dir / "suspect_nodes.csv", index=False)

    print("[5/6] Vérifications...")
    assert min(G.nodes) == 0
    assert max(G.nodes) == G.number_of_nodes() - 1
    assert not any(u == v for u, v in G.edges)
    assert nx.is_connected(G)
    assert metadata["suspicion_score"].notna().all(), "score manquant pour certains nœuds"
    # La renumérotation respecte l'ordre des identifiants d'origine.
    assert list(map(int, mapping_df["original_id"])) == sorted(map(int, mapping_df["original_id"]))

    print("[6/6] Terminé.")
    print(f"    Graphe       : {graph_path}")
    print(f"    Mapping      : {output_dir / 'node_mapping.csv'}")
    print(f"    Métadonnées  : {output_dir / 'nodes_metadata.csv'}")
    print(f"    À vérifier   : {output_dir / 'suspect_nodes.csv'}")
    print(
        "    Remarque : les nœuds du graphml sont relus sous forme de chaînes "
        "('0', '1', ...) ; les convertir en entiers au chargement."
    )


def main():
    parser = argparse.ArgumentParser(description="Nettoyage du graphe des aéroports.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    clean_graph(args.input, args.output)


if __name__ == "__main__":
    main()