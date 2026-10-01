"""
Registre des méthodes : ajouter ici chaque nouvelle méthode.
Le nom sert dans la configuration : method: <nom>
"""

from methods.distance import DistanceOnly
from methods.heuristics import AdamicAdar, CommonNeighbors, PreferentialAttachment

METHODS = {
    "distance": DistanceOnly,
    "common_neighbors": CommonNeighbors,
    "adamic_adar": AdamicAdar,
    "pref_attachment": PreferentialAttachment,
}

# Le GAE nécessite PyTorch et PyTorch Geometric : on ne le charge que s'ils sont installés
try:
    from methods.gae import GAE
    METHODS["gae"] = GAE
except ImportError:
    pass