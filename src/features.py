"""MEMBRE 1 - Construction des features des noeuds."""
import torch as th
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder


def build_features(info, coord=True, pop=True, ctry=True):

    n = len(info["names"])
    feats = []
    if coord:
        la, lo = th.deg2rad(info["lat"]), th.deg2rad(info["lon"])
        feats.append(th.stack([th.cos(la) * th.cos(lo),
                               th.cos(la) * th.sin(lo),
                               th.sin(la)], dim=1))
    if pop:
        lp = th.log(info["pop"])
        feats.append(((lp - lp.mean()) / lp.std()).unsqueeze(1))
    if ctry:
        c = th.tensor(LabelEncoder().fit_transform(info["country"]), dtype=th.long)
        feats.append(F.one_hot(c).float())
    if len(feats) == 0:
        return th.ones((n, 1))
    return th.cat(feats, dim=1)


def build_coords(info):

    return build_features(info, coord=True, pop=False, ctry=False)