import networkx as nx
import torch as th
from torch_geometric.data import Data


def load_graph(path):
    return nx.read_graphml(path, node_type=int)


def graph_to_tensors(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}  # numero de noeud -> indice 0..N-1

    edges = [(idx[u], idx[v]) for u, v in G.edges() if u != v]
    edge_index = th.tensor(edges, dtype=th.long).t()
    edge_index = th.cat([edge_index, edge_index.flip(0)], dim=1)

    info = {
        "lon": th.tensor([G.nodes[n]["lon"] for n in nodes], dtype=th.float),
        "lat": th.tensor([G.nodes[n]["lat"] for n in nodes], dtype=th.float),
        "pop": th.tensor([G.nodes[n]["population"] for n in nodes], dtype=th.float),
        "country": [G.nodes[n]["country"] for n in nodes],
        "names": [G.nodes[n]["city_name"] for n in nodes],
    }
    return edge_index, info


def make_data(x, edge_index):
    return Data(x=x, edge_index=edge_index)
