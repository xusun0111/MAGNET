import torch as pt
import torch.nn as nn
import torch_geometric as pyg
from load_dataset import load_dataset, compute_second_order_neighbors
import GCL
from GCL.eval import get_split, LREvaluator
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from tqdm import tqdm
import warnings
import torch
import torch_geometric
import time
import argparse

class FDE_encoder(nn.Module):
    def __init__(self, n_in, n_h, n_out, activator):
        super(FDE_encoder, self).__init__()
        self.act = activator()
        self.lin_in = nn.Linear(n_in, n_h)
        self.lin_h1 = nn.Linear(n_h, n_h)
        self.lin_out = nn.Linear(n_h, n_out)
        self.sample = []

    def encode(self, x):
        z = self.act(self.lin_in(x))
        z = self.act(self.lin_h1(z))
        return self.lin_out(z)

    def forward(self, x: pt.Tensor):
        self.sample = x
        self.out = F.normalize(self.encode(x.T))
        return self.out

    def FDE_loss(self):
        mean_loss = self.out.mean(dim=0).pow(2).mean()
        var_loss = -self.out.var(dim=0).mean()
        return mean_loss + 0.1 * var_loss

class GCN_encoder(nn.Module):
    def __init__(self, n_in, n_h, activator):
        super(GCN_encoder, self).__init__()
        self.gcn_in = GCNConv(n_in, n_h)
        self.gcn_out = GCNConv(n_h, n_h)
        self.act = activator()

    def encode(self, x, edge_index):
        out = self.act(self.gcn_in(x, edge_index))
        out = self.gcn_out(out, edge_index)
        return out

    def proj(self, z):
        return self.lin_2(self.act(self.lin_1(z)))

    def forward(self, x, edge_index):
        out = self.encode(x, edge_index)
        return out

    def embed(self, x, edge_index):
        self.eval()
        return self.encode(x, edge_index)

class GraphTransformer_encoder(nn.Module):
    def __init__(self, n_in, n_h, activator, heads=3, dropout=0.2):
        super(GraphTransformer_encoder, self).__init__()
        self.transformer_in = TransformerConv(n_in, n_h, heads=heads, dropout=dropout)
        self.transformer_out = TransformerConv(n_h * heads, n_h, heads=heads, dropout=dropout)
        self.act = activator()

    def encode(self, x, edge_index):
        out = self.act(self.transformer_in(x, edge_index))
        out = self.act(self.transformer_out(out, edge_index))
        return out

    def proj(self, z):
        return self.lin_2(self.act(self.lin_1(z)))

    def forward(self, x, edge_index):
        out = self.encode(x, edge_index)
        return out

    def embed(self, x, edge_index):
        self.eval()
        return self.encode(x, edge_index)

class MAGNET(nn.Module):
    def __init__(self, D_NN, GCN, Trans, S_mtd, sample_size):
        super(MAGNET, self).__init__()
        self.dnn = D_NN
        self.gcn = GCN
        self.trans = Trans
        self.trans_proj = nn.Linear(3072, 1024)
        self.smtd = S_mtd
        self.sample_size = sample_size
        self.d_sample_matrix = []

    def update_sample(self, x, edge_index, if_rand=False):
        with torch.no_grad():
            self.d_sample_matrix = self.smtd(self.sample_size, x, edge_index, if_rand)

    def forward(self, x, edge_index):
        dimension_sig = self.dnn(self.d_sample_matrix)
        x = self.feature_sig_propagate(x, dimension_sig)
        z_g = self.gcn(x, edge_index)
        z_t = self.trans(x, edge_index)
        return z_g, z_t

    def embed(self, x, edge_index):
        with torch.no_grad():
            self.eval()
            dimension_sig = self.dnn(self.d_sample_matrix)
            x = self.feature_sig_propagate(x, dimension_sig)
            z_g = self.gcn.embed(x, edge_index)
            z_t = self.trans.embed(x, edge_index)
            # z_t = self.trans_proj(z_t)
            z = torch.cat((z_g, z_t), dim=1)
            return z

    def loss_neg(self, z):
        z = F.normalize(z, dim=1)
        return z.mean(dim=0).pow(2).mean()

    def loss_pos(self, z, edge_index):
        if edge_index.numel() == 0:
            return torch.tensor(0.0, device=z.device)
        return (z[edge_index[0]] - z[edge_index[1]]).pow(2).mean()

    def loss_fde(self):
        return self.dnn.FDE_loss()

    def feature_sig_propagate(self, x, dimension_sig):
        return F.normalize(x @ dimension_sig)
