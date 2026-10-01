"""
Interface commune : toutes les méthodes (heuristiques, MLP, GNN) la respectent.
"""

import numpy as np


class Method:
    name = "base"

    def __init__(self, **config):
        self.config = config

    def fit(self, split: dict, meta) -> "Method":
        """
        Apprend à partir de split["train_pos"] uniquement.
        split["val_pos"], split["val_neg_*"] peuvent servir à l'early stopping.
        Ne jamais utiliser split["test_*"] ici.
        """
        return self

    def score(self, pairs: np.ndarray) -> np.ndarray:
        """Retourne un score par paire (P,), plus haut = route plus probable."""
        raise NotImplementedError
