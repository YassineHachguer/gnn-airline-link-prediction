"""
Graph Auto-Encoder (GAE) pour la prédiction de liens.

- Encodeur GNN (GCN, GraphSAGE ou GAT) : calcule un vecteur z par aéroport
  à partir de ses features et de ses voisins (message passing).
- Décodeur produit scalaire : score(u, v) = z_u · z_v.
- Entraînement : routes d'entraînement (positifs) contre paires aléatoires
  non reliées (négatifs), perte d'entropie croisée binaire.
- Early stopping sur l'AUC de validation.

Exemples :
    python src/run.py --config configs/gae_gcn.yaml --eval val
    python src/run.py --config configs/gae_sage.yaml --eval val

Les features de paires (bloc "pair" de la config) ne sont pas utilisées ici :
le décodeur est un simple produit scalaire. Elles serviront au modèle proposé.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling

from features import DEFAULT_CONFIG, build_node_features
from methods.base import Method


DEFAULTS = {
    "encoder": "gcn",     # gcn, sage ou gat
    "layers": 2,
    "hidden": 64,
    "out": 32,            # dimension de z
    "dropout": 0.3,
    "lr": 0.01,
    "epochs": 300,
    "patience": 30,       # arrêt si l'AUC de validation ne progresse plus
    "seed": 0,
    "features": DEFAULT_CONFIG,
}

CONVS = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}


class Encoder(torch.nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        conv = CONVS[cfg["encoder"]]
        dims = [in_dim] + [cfg["hidden"]] * (cfg["layers"] - 1) + [cfg["out"]]
        self.convs = torch.nn.ModuleList(conv(dims[i], dims[i + 1]) for i in range(cfg["layers"]))
        self.dropout = cfg["dropout"]

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # pas d'activation sur la dernière couche
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x


def to_edge_index(edges):
    # Graphe non orienté : chaque arête dans les deux sens pour le message passing
    e = torch.as_tensor(edges, dtype=torch.long).t()
    return torch.cat([e, e.flip(0)], dim=1)


def dot(z, pairs):
    return (z[pairs[0]] * z[pairs[1]]).sum(dim=-1)


class GAE(Method):
    name = "gae"

    def __init__(self, **config):
        super().__init__(**config)
        self.cfg = {**DEFAULTS, **config}

    def fit(self, split, meta):
        cfg = self.cfg
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        x, _ = build_node_features(meta, split["train_pos"], cfg["features"])  # degré sur train uniquement
        self.x = torch.as_tensor(x)
        self.edge_index = to_edge_index(split["train_pos"])  # seules les routes d'entraînement
        train_pos = torch.as_tensor(split["train_pos"], dtype=torch.long).t()

        self.model = Encoder(self.x.shape[1], cfg)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])

        val_pos = torch.as_tensor(split["val_pos"], dtype=torch.long).t()
        val_neg = torch.as_tensor(split["val_neg_random"], dtype=torch.long).t()
        y_val = np.r_[np.ones(val_pos.shape[1]), np.zeros(val_neg.shape[1])]

        best_auc, best_state, wait = -1.0, None, 0
        for epoch in range(cfg["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            z = self.model(self.x, self.edge_index)

            # Nouveaux négatifs aléatoires à chaque époque
            neg = negative_sampling(self.edge_index, num_nodes=len(meta), num_neg_samples=train_pos.shape[1])
            pos_logits, neg_logits = dot(z, train_pos), dot(z, neg)
            loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) \
                 + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            loss.backward()
            optimizer.step()

            # Early stopping sur la validation
            self.model.eval()
            with torch.no_grad():
                z = self.model(self.x, self.edge_index)
                scores = torch.cat([dot(z, val_pos), dot(z, val_neg)]).numpy()
            auc = roc_auc_score(y_val, scores)
            if auc > best_auc:
                best_auc, best_state, wait = auc, copy.deepcopy(self.model.state_dict()), 0
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        with torch.no_grad():
            self.z = self.model(self.x, self.edge_index)
        print(f"  arrêt à l'époque {epoch}, meilleure AUC val = {best_auc:.3f}")
        return self

    def score(self, pairs):
        pairs = torch.as_tensor(pairs, dtype=torch.long).t()
        with torch.no_grad():
            return dot(self.z, pairs).numpy()
