"""
Baseline « distance seule » : plus deux aéroports sont proches, plus la route est probable.
Aucun apprentissage. Teste H1.
"""

import numpy as np

from features import haversine_km
from methods.base import Method


class DistanceOnly(Method):
    name = "distance"

    def fit(self, split, meta):
        self.lat = meta["lat"].to_numpy()
        self.lon = meta["lon"].to_numpy()
        return self

    def score(self, pairs):
        u, v = pairs[:, 0], pairs[:, 1]
        return -haversine_km(self.lat[u], self.lon[u], self.lat[v], self.lon[v])
