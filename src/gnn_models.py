import torch as th
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GAE, VGAE, GATConv, GCNConv, SAGEConv


# ici on factorise : on choisit le type de couche avec un parametre
def make_conv(kind, in_dim, out_dim):
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "sage":
        return SAGEConv(in_dim, out_dim)
    if kind == "gat":
        return GATConv(in_dim, out_dim, heads=1)  # 1 tete pour garder la meme dim
    raise ValueError(f"conv inconnue : {kind}")


# encodeur parametrable pour l ablation (nb de couches, VGAE ou GAE, type de conv)
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, variational=True, conv="gcn"):
        super().__init__()
        self.n_layers = n_layers
        self.variational = variational
        if n_layers == 2:
            self.conv1 = make_conv(conv, in_dim, hidden)
            last_in = hidden
        else:
            last_in = in_dim
        self.conv_mu = make_conv(conv, last_in, out_dim)
        if variational:
            self.conv_logstd = make_conv(conv, last_in, out_dim)  # seulement pour le VGAE

    def forward(self, x, edge_index):
        if self.n_layers == 2:
            x = F.relu(self.conv1(x, edge_index))
        if self.variational:
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        return self.conv_mu(x, edge_index)


# decodeur MLP a la place du produit scalaire
# symetrique : score(u,v) = score(v,u)
# option : ajouter la distance entre u et v (info sur la paire, pas sur une seule ville)
class MLPDecoder(nn.Module):
    def __init__(self, z_dim, hidden=32, coords=None):
        super().__init__()
        self.use_dist = coords is not None
        if self.use_dist:
            self.register_buffer("coords", coords)
        in_dim = 2 * z_dim + (1 if self.use_dist else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, edge_index, sigmoid=True):
        zu, zv = z[edge_index[0]], z[edge_index[1]]
        h = [zu * zv, (zu - zv).abs()]
        if self.use_dist:
            cu, cv = self.coords[edge_index[0]], self.coords[edge_index[1]]
            h.append((cu - cv).norm(dim=1, keepdim=True))
        s = self.net(th.cat(h, dim=1)).squeeze(-1)
        return th.sigmoid(s) if sigmoid else s


# construit le modele complet : chaque experience = un appel a build_model
def build_model(in_dim, hidden=64, out_dim=32, n_layers=2, variational=True,
                conv="gcn", decoder="dot", coords=None):
    enc = Encoder(in_dim, hidden, out_dim, n_layers, variational, conv)
    if decoder == "dot":
        dec = None  # produit scalaire par defaut de PyG
    elif decoder == "mlp":
        dec = MLPDecoder(out_dim)
    elif decoder == "mlp_dist":
        if coords is None:
            raise ValueError("le decodeur 'mlp_dist' necessite coords")
        dec = MLPDecoder(out_dim, coords=coords)
    else:
        raise ValueError(f"decodeur inconnu : {decoder}")
    return VGAE(enc, dec) if variational else GAE(enc, dec)