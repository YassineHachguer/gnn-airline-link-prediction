"""
Heuristiques classiques de prédiction de liens (sans apprentissage).
Calculées uniquement sur le graphe d'entraînement.

- Common Neighbors      : nombre de voisins communs
- Adamic-Adar           : voisins communs pondérés par 1 / log(degré) (les petits voisins comptent plus)
- Preferential Attachment : deg(u) * deg(v) (les hubs se relient entre eux)
"""

import numpy as np
import scipy.sparse as sp

from methods.base import Method


def train_adjacency(split, n_nodes):
    edges = split["train_pos"]
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


class CommonNeighbors(Method):
    name = "common_neighbors"

    def fit(self, split, meta):
        A = train_adjacency(split, len(meta))
        self.M = (A @ A).tocsr()
        return self

    def score(self, pairs):
        return np.asarray(self.M[pairs[:, 0], pairs[:, 1]]).ravel()


class AdamicAdar(Method):
    name = "adamic_adar"

    def fit(self, split, meta):
        A = train_adjacency(split, len(meta))
        deg = np.asarray(A.sum(axis=1)).ravel()
        # Un voisin commun de degré 1 ne peut pas exister (il relierait 2 nœuds) : log(deg) > 0
        weights = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
        self.M = (A @ sp.diags(weights) @ A).tocsr()
        return self

    def score(self, pairs):
        return np.asarray(self.M[pairs[:, 0], pairs[:, 1]]).ravel()


class PreferentialAttachment(Method):
    name = "pref_attachment"

    def fit(self, split, meta):
        A = train_adjacency(split, len(meta))
        self.deg = np.asarray(A.sum(axis=1)).ravel()
        return self

    def score(self, pairs):
        return self.deg[pairs[:, 0]] * self.deg[pairs[:, 1]]
