import copy
import networkx as nx
import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------- Structure seule : Adamic-Adar ----------
def adamic_adar(splits):
    _, _, test_d = splits
    g = nx.Graph()
    g.add_nodes_from(range(test_d.num_nodes))
    g.add_edges_from(test_d.edge_index.t().tolist())
    pos = test_d.pos_edge_label_index.t().tolist()
    neg = test_d.neg_edge_label_index.t().tolist()
    s_pos = [p for _, _, p in nx.adamic_adar_index(g, pos)]
    s_neg = [p for _, _, p in nx.adamic_adar_index(g, neg)]
    y = [1] * len(s_pos) + [0] * len(s_neg)
    s = s_pos + s_neg
    return roc_auc_score(y, s), average_precision_score(y, s)


# ---------- Features seules : MLP sans graphe ----------
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


def _auc_ap(z, pos, neg):
    with th.no_grad():
        s = th.cat([(z[pos[0]] * z[pos[1]]).sum(1),
                    (z[neg[0]] * z[neg[1]]).sum(1)]).sigmoid().numpy()
    y = np.r_[np.ones(pos.shape[1]), np.zeros(neg.shape[1])]
    return roc_auc_score(y, s), average_precision_score(y, s)


def run_mlp(splits, seed, epochs=200, hidden=64, out=32, lr=0.01):
    train_d, val_d, test_d = splits
    x = train_d.x
    th.manual_seed(seed)
    enc = MLPEncoder(x.shape[1], hidden, out)
    opt = th.optim.Adam(enc.parameters(), lr=lr)
    pos = train_d.pos_edge_label_index

    best_auc, best_state = -1, None
    for _ in range(epochs):
        enc.train()
        opt.zero_grad()
        z = enc(x)
        neg = negative_sampling(train_d.edge_index, num_nodes=x.shape[0],
                                num_neg_samples=pos.shape[1])
        ps = (z[pos[0]] * z[pos[1]]).sum(1)
        ns = (z[neg[0]] * z[neg[1]]).sum(1)
        loss = F.binary_cross_entropy_with_logits(ps, th.ones_like(ps)) + \
               F.binary_cross_entropy_with_logits(ns, th.zeros_like(ns))
        loss.backward()
        opt.step()

        enc.eval()
        val_auc, _ = _auc_ap(enc(x), val_d.pos_edge_label_index, val_d.neg_edge_label_index)
        if val_auc > best_auc:
            best_auc, best_state = val_auc, copy.deepcopy(enc.state_dict())

    enc.load_state_dict(best_state)
    enc.eval()
    return _auc_ap(enc(x), test_d.pos_edge_label_index, test_d.neg_edge_label_index)
