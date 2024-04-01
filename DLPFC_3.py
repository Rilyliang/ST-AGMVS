from __future__ import division
from __future__ import print_function
import os
import argparse
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
from scipy.spatial import distance
import torch
from torch.optim import Adam
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from ST_model_3.utils import *
from ST_model_3.train_model_3 import *
import warnings
warnings.filterwarnings('ignore')

ari_avg = 0.0
nmi_avg = 0.0
# data_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151674', '151675', '151676']
data_list = ['151507']
for i in data_list:
    fix_seed(42)
    data_path = 'D:/scientific study/data/82_spatialLIBD_LIBD human dorsolateral pre-frontal cortex (DLPFC) spatial transcriptomics data generated with the 10x Genomics Visium platform/'
    section_name = i
    print(section_name)
    adata = read_10X_DLPFC(data_path, section_name)
    adata = normalize(adata, highly_genes=3000)

    sadj = spatial_construct_graph(adata, radius=560)
    fadj = features_construct_graph(adata, k=14)

    labels = adata.obs['ground']
    _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
    n = len(ground)
    class_num = len(np.unique(ground))

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    st_model = Trainer_3()
    ARI, NMI, ARI_refined, NMI_refined = st_model.train(
        adata,
        sadj,
        fadj,
        ground,
        device,
        class_num,
        nhid1=1000,
        nhid2=512,
        nhid3=128,
        nz=64,
        nout=32,
        dropout=0.1,
        max_epochs=200,
        lr=0.001,
        weight_decay=5e-4,
        )

    print(section_name, ' ', ARI)
    print(section_name, ' ', NMI)
    print("refined", section_name, ' ', ARI_refined)
    print("refined", section_name, ' ', NMI_refined)

    ari_avg = ari_avg + ARI
    nmi_avg = nmi_avg + NMI

    savepath = 'D:/scientific study/paper3/ST-GCAA/' + section_name + '/'
    try:
        os.mkdir(savepath)
        print(f"文件夹创建成功")
    except FileExistsError:
        print(f"文件夹已经存在")
    except Exception as e:
        print(f"无法创建文件夹': {e}")

    adata.write(savepath + 'ST-GCAA_sample_data.h5ad')

    with open(savepath + "ari_and_nmi.txt", "w") as file:
        file.write("ARI: {}\n".format(ARI))
        file.write("NMI: {}\n".format(NMI))
        file.write("ARI_refined: {}\n".format(ARI_refined))
        file.write("NMI_refined: {}\n".format(NMI_refined))

    title = 'ST-GCAA: ARI={:.3f}, NMI={:.3f}'.format(adata.uns['ARI_refine'], adata.uns['NMI_refine'])
    sc.pl.spatial(adata, img_key="hires", color=['domain'], title=title, show=False)
    plt.savefig(savepath + 'ST-GCAA.tif', bbox_inches='tight', dpi=300)

    sc.pp.neighbors(adata, use_rep='emb')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.tl.paga(adata, groups='domain')
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2,
                       show=False)
    plt.savefig(savepath + 'ST-GCAA_PAGA_domain.tif', bbox_inches='tight', dpi=300)

    sc.tl.paga(adata, groups='ground_truth')
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2,
                       show=False)
    plt.savefig(savepath + 'ST-GCAA_PAGA_ground_truth.tif', bbox_inches='tight', dpi=300)

print("ARI_AVG: ", ari_avg/12.0)
print("NMI_AVG: ", nmi_avg/12.0)


