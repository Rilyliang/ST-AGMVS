import scipy.sparse as sp
import sklearn
import torch
import torch.nn as nn
import networkx as nx
from sklearn.cluster import KMeans
import torch.nn.functional as F
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from sklearn.metrics.pairwise import euclidean_distances
import random
import os
from torch.backends import cudnn
import math
from sklearn.neighbors import NearestNeighbors
import ot
import torch
import scipy.sparse as sp
import anndata as ad
def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000  ########
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata

def normalize_1(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata) ##############
    sc.pp.scale(adata, zero_center=False, max_value=10)

    return adata

def read_10X_DLPFC(data_path, section_name):

    labels_path = data_path + section_name + '/metadata.tsv'
    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)

    adata1 = sc.read_visium(data_path + section_name, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()
    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']
    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)

    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))
    adata.var_names_make_unique()
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.obs['array_row'] = adata1.obs['array_row']
    adata.obs['array_col'] = adata1.obs['array_col']
    adata.uns['spatial'] = adata1.uns['spatial']
    adata.var['gene_ids'] = adata1.var['gene_ids']
    adata.var['feature_types'] = adata1.var['feature_types']
    adata.var['genome'] = adata1.var['genome']
    adata.var_names_make_unique()

    return adata

def read_Human_breast_cancer(data_path):

    adata = sc.read_visium(data_path, count_file='V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5',load_images=True)
    adata.var_names_make_unique()
    df_meta = pd.read_csv(data_path + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['ground_truth']
    adata.obs['ground'] = df_meta_layer.values
    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground'])]
    for i in range(adata.shape[0]):
        if adata.obs['ground'][i] == 'IDC_8':
            adata.obs['ground'][i] = 'DCIS/LCIS_3'

    return adata


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    position = position.astype(np.float)  ##########我加的
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type

def spatial_construct_graph(adata, radius=150):

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0]))

    # print("coor:", coor)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]]=1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0],coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)

    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()

    return sadj

def features_construct_graph(adata, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    features = adata.X
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)

    adata.obsm['fadj'] = fadj

    return fadj

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):###定义ClusterLoss这个类的基本参数和方法
        super(ClusterLoss, self).__init__()
        self.class_num = class_num###类别数目，比如deng是10
        self.temperature = temperature###温度参数=1.0
        self.mask = self.mask_correlated_clusters(class_num)###定义mask的方式
        self.criterion = nn.CrossEntropyLoss(reduction="sum")###定义损失标准采用交叉熵
        self.similarity_f = nn.CosineSimilarity(dim=2)###定义相似性采用余弦相似度

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))###生成(N, N)的全1矩阵
        mask = mask.fill_diagonal_(0)###对角线元素置0
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()###转成bool型变量，即其中的1变成True
        return mask

    def forward(self, c_i, c_j):##对每一批，c_i是256个10维的向量，
        p_i = c_i.sum(0).view(-1)###把这一批256个求和，得到一个总的p_i，是一个10维的向量
        p_i /= p_i.sum()###p_i.sum()=256,所以p_i这里是求平均,是一个10维的向量
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()###log(p_i)求得是以e为底的ln(p_i),
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()####转置，变成10*256的
        c_j = c_j.t()###转置，变成10*256的
        N = 2 * self.class_num###N=20
        c = torch.cat((c_i, c_j), dim=0)##拼接，变成20*256的

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature###20*20的
        sim_i_j = torch.diag(sim, self.class_num)###取矩阵的主对角线元素 1*10
        sim_j_i = torch.diag(sim, -self.class_num) #取矩阵的副对角线元素 1*10

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) #20 * 1
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss