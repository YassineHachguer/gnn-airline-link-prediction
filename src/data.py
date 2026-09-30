import html
import io
import re
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops

XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}


def _read_graphml_robust(path: str) -> nx.Graph:
    txt = open(path, encoding="utf-8").read()
    txt = txt[txt.index("<graphml"):]

    def fix(m):
        name = m.group(1)
        return m.group(0) if name in XML_ENTITIES else html.unescape(m.group(0))

    txt = re.sub(r"&(\w+);", fix, txt)
    return nx.read_graphml(io.BytesIO(txt.encode("utf-8")))


def _clean_country(c: str) -> str:
    return c.split("#")[-1].split(",")[0].strip().upper()


def load_airports(path: str = "data/airports.graphml") -> Data:
    G = _read_graphml_robust(path)
    nodes = sorted((n for n in G.nodes if "lat" in G.nodes[n]), key=int)
    idx = {n: i for i, n in enumerate(nodes)}

    lat = np.array([G.nodes[n]["lat"] for n in nodes], dtype=np.float32)
    lon = np.array([G.nodes[n]["lon"] for n in nodes], dtype=np.float32)
    pop = np.array([G.nodes[n]["population"] for n in nodes], dtype=np.float32)
    countries = [_clean_country(G.nodes[n]["country"]) for n in nodes]
    names = [G.nodes[n]["city_name"] for n in nodes]

    la, lo = np.radians(lat), np.radians(lon)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)

    country2idx = {c: i for i, c in enumerate(sorted(set(countries)))}
    y = torch.tensor([country2idx[c] for c in countries], dtype=torch.long)

    edges = [(idx[u], idx[v]) for u, v in G.edges if u in idx and v in idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)  

    data = Data(
        x=torch.tensor(np.c_[xyz, np.log10(pop)], dtype=torch.float), 
        edge_index=edge_index,
        y=y,
    )
    data.lat = torch.tensor(lat)
    data.lon = torch.tensor(lon)
    data.pop = torch.tensor(pop)
    data.pop_missing = torch.tensor(pop == 10000, dtype=torch.float) 
    data.names = names
    data.countries = countries
    data.country2idx = country2idx
    return data


if __name__ == "__main__":
    d = load_airports()
    print(d)
    print("non orienté :", d.is_undirected())
    print("arêtes non orientées :", d.edge_index.size(1) // 2)
    print("populations manquantes (=10000) :", int(d.pop_missing.sum()))
    print("pays distincts :", len(d.country2idx))