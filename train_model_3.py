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
from scipy.sparse import csc_matrix, csr_matrix
from sklearn import metrics
from ST_model_3.model_3_set import *
from ST_model_3.utils import *
from ST_model_3.ZINB_layer import *

class Trainer_3(object):
    def __init__(self):
        super(Trainer_3, self).__init__()

    def train(self,
              adata,
              sadj,
              fadj,
              ground,
              device,
              n_clusters,
              nhid1,
              nhid2,
              nhid3,
              nz,
              nout,
              dropout,
              max_epochs,
              lr,
              weight_decay,
              radius=50,
              refinement=True
              ):
        self.device = device
        self.n_clusters = n_clusters
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nhid3 = nhid3
        self.nz = nz
        self.nout = nout
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.radius = radius
        self.refinement = refinement

        assert adata.shape[0] == sadj.shape[0] == sadj.shape[1] == fadj.shape[0] == fadj.shape[1]

        adata_Vars = adata[:, adata.var['highly_variable']]

        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        sadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
        sadj = sparse_mx_to_torch_sparse_tensor(sadj)
        sadj = sadj.to(self.device)

        fadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
        fadj = sparse_mx_to_torch_sparse_tensor(fadj)
        fadj = fadj.to(self.device)

        features = torch.FloatTensor(feat)
        features = features.to(self.device)

        self.model = Spatial_model_3(
            nfeat=feat.shape[1],
            nhid1=self.nhid1,
            nhid2=self.nhid2,
            nhid3=self.nhid3,
            nz=self.nz,
            nout=self.nout,
            nclusters=self.n_clusters
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        ari_max = 0.0
        nmi_max = 0.0
        idx_max = []
        mean_max = []
        emb_max = []
        for epoch in range(self.max_epochs):
            self.model.train()
            optimizer.zero_grad()

            # ST-GCAA 损失
            emb, pi, disp, mean, sc1, fc1, x_bar = self.model(features, sadj, fadj)
            zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
            re_loss = F.mse_loss(x_bar, features)
            cluster_loss = ClusterLoss(self.n_clusters, 0.6)
            c_loss = cluster_loss(sc1, fc1)
            total_loss = 1 * zinb_loss + 0.5 * re_loss + 1 * c_loss

            # ST-GCAA 消除自动编码器后的损失
            # emb, pi, disp, mean, sc1, fc1 = self.model(features, sadj, fadj)
            # zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
            # # re_loss = F.mse_loss(x_bar, features)
            # cluster_loss = ClusterLoss(self.n_clusters, 0.6)
            # c_loss = cluster_loss(sc1, fc1)
            # total_loss = 1 * zinb_loss + 1 * c_loss

            # # # ST-GCAA 只使用特征邻域结构的损失
            # emb, pi, disp, mean, x_bar = self.model(features, sadj, fadj)
            # zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
            # re_loss = F.mse_loss(x_bar, features)
            # # cluster_loss = ClusterLoss(self.n_clusters, 0.6)
            # # c_loss = cluster_loss(sc1, fc1)
            # total_loss = 1 * zinb_loss + 0.5 * re_loss

            emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
            mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values

            print('epoch: ', epoch,
                  ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' re_loss = {:.2f}'.format(re_loss),
                  ' c_loss = {:.2f}'.format(c_loss),
                  # ' instance_loss = {:.2f}'.format(inst_loss),
                  ' total_loss = {:.2f}'.format(total_loss))

            total_loss.backward()
            optimizer.step()

            kmeans = KMeans(n_clusters=self.n_clusters).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(ground, idx)
            nmi = metrics.normalized_mutual_info_score(ground, idx)

            if ari_res > ari_max:
                ari_max = ari_res
                nmi_max = nmi
                idx_max = idx
                mean_max = mean
                emb_max = emb

        adata.obs['K-means'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.uns['ARI'] = ari_max
        adata.uns['NMI'] = nmi_max

        if self.refinement:
            new_type = refine_label(adata, self.radius, key='K-means')
            adata.obs['domain'] = new_type

        ARI_refine = metrics.adjusted_rand_score(ground, new_type)
        NMI_refine = metrics.normalized_mutual_info_score(ground, new_type)
        adata.uns['ARI_refine'] = ARI_refine
        adata.uns['NMI_refine'] = NMI_refine

        return ari_max, nmi_max, ARI_refine, NMI_refine


