from __future__ import print_function, division
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 一层GCN
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

class AE(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nz):
        super(AE, self).__init__()
        self.nfeat = nfeat
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nhid3 = nhid3
        self.nz = nz

        self.en1 = Linear(self.nfeat, self.nhid1)
        self.en2 = Linear(self.nhid1, self.nhid2)
        self.en3 = Linear(self.nhid2, self.nhid3)
        self.z_layer = Linear(self.nhid3, self.nz)

        self.den1 = Linear(self.nz, self.nhid3)
        self.den2 = Linear(self.nhid3, self.nhid2)
        self.den3 = Linear(self.nhid2, self.nhid1)
        self.x_bar_layer = Linear(self.nhid1, self.nfeat)

    def forward(self, x):
        en_h1 = F.relu(self.en1(x))
        en_h2 = F.relu(self.en2(en_h1))
        en_h3 = F.relu(self.en3(en_h2))
        z = self.z_layer(en_h3)

        den_h1 = F.relu(self.den1(z))
        den_h2 = F.relu(self.den2(den_h1))
        den_h3 = F.relu(self.den3(den_h2))
        x_bar = self.x_bar_layer(den_h3)

        return en_h1, en_h2, en_h3, z, x_bar

class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class Spatial_model_3(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nz, nout, nclusters):
        super(Spatial_model_3, self).__init__()
        self.nfeat = nfeat
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nhid3 = nhid3
        self.nz = nz
        self.nout = nout
        self.nclusters = nclusters

        self.ae_encoder = AE(
            self.nfeat,
            self.nhid1,
            self.nhid2,
            self.nhid3,
            self.nz
        )

        self.gcn1 = GCNLayer(self.nfeat, self.nhid1)
        self.gcn2 = GCNLayer(self.nhid1, self.nhid2)
        self.gcn3 = GCNLayer(self.nhid2, self.nhid3)
        self.gcn4 = GCNLayer(self.nhid3, self.nz)
        self.gcn5 = GCNLayer(self.nz, self.nout)

        self.ZINB = decoder(self.nfeat, self.nhid1, self.nout)
        self.att = Attention(self.nout)
        self.MLP = nn.Sequential(
            nn.Linear(self.nout, self.nout)
        )
        self.cluster_projector = nn.Sequential(  #####聚类投影，z_dim维投影到n_clusters维
            nn.Linear(self.nout, self.nclusters),  ####32-10
            nn.Softmax(dim=1))

    def forward(self, x, sadj, fadj):


