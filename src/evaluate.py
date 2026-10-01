"""
Métriques d'évaluation communes à toutes les méthodes.

Chaque méthode renvoie un score par paire (plus haut = route plus probable).
On compare les scores des positifs (vraies routes cachées) à ceux des négatifs.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def hits_at_k(pos_scores, neg_scores, k=50):
    """Part des positifs dont le score dépasse le k-ième meilleur négatif."""
    if len(neg_scores) < k:
        return float("nan")
    threshold = np.sort(neg_scores)[-k]
    return float((pos_scores > threshold).mean())


def evaluate(pos_scores, neg_scores, k=50):
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        f"hits@{k}": hits_at_k(pos_scores, neg_scores, k),
    }
