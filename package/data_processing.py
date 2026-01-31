import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

from package.utils import read_json, filter_specialgenes, gaussian_smooth_data, construct_adjacency_matrix


class Load10xAdata:
    def __init__(self, path, dataset, slice,config,args,adata=None):
        self.path = path
        self.dataset=dataset
        self.slice=slice
        self.label = args.label
        self.min_cells=config['min_cells']
        self.min_counts=config['min_counts']
        self.num_neighbors=config['num_neighbors']
        self.num_pruning=config['num_pruning']
        self.fdim = config['fdim']
        self.smooth_r=config['smooth_r']
        self.adata = adata
        self.d = read_json(os.path.join(self.path, 'spatial', 'scalefactors_json.json'))

    def load_data(self):
        print('Load data!')
        self.adata = sc.read_visium(self.path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        self.adata.var_names_make_unique()

    def preprocess(self):
        print('Preprocess!')
        sc.pp.filter_genes(self.adata, min_cells=self.min_cells)
        sc.pp.filter_genes(self.adata, min_counts=self.min_counts)
        filter_specialgenes(self.adata)
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.fdim)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.scale(self.adata, zero_center=False, max_value=10)


    def generate_gene_expression(self):
        print('Generate gene expression!')
        adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        self.adata=adata_Vars
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        self.adata.obsm['gene_feat'] = feat



    def load_label(self):
        print('Load label!')
        df_meta = pd.read_csv(os.path.join(self.path, 'truth.txt'), sep='\t', header=None)
        df_meta_layer = df_meta[1]

        self.adata.obs['ground_truth'] = df_meta_layer.values

        self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]



    def run(self):
        self.load_data()
        self.preprocess()
        self.generate_gene_expression()
        if self.label:
            self.load_label()


        self.adata.uns['d'] = self.d
        self.adata.uns['smooth_r'] = self.smooth_r*self.d
        self.adata=gaussian_smooth_data(self.adata)
        self.adata=construct_adjacency_matrix(self.adata,self.num_neighbors,self.num_pruning)
        self.adata.write('./results/'+self.dataset+'/'+self.slice+'/data_processing.h5ad')
        print('adata load done!')

        return self.adata



class LoadAdata:
    def __init__(self, path, dataset, slice,config,args,adata=None,used_barcode=False,d=2):
        self.path = path
        self.dataset=dataset
        self.slice=slice
        self.label = args.label
        self.platform=args.platform
        self.used_barcode=used_barcode
        self.d=d
        self.fdim = config['fdim']
        self.smooth_r=config['smooth_r']
        self.min_cells=config['min_cells']
        self.min_counts=config['min_counts']
        self.num_neighbors=config['num_neighbors']
        self.num_pruning=config['num_pruning']
        self.adata = adata

    def load_data(self):
        print('Load data!')
        if self.dataset == 'Mouse_Olfactory_Bulb':
            if not os.path.exists(os.path.join(self.path, 'raw.h5ad')):
                counts=pd.read_csv(os.path.join(self.path,'RNA_counts.tsv'),sep='\t',index_col=0).T
                counts.index = [f'Spot_{i}' for i in counts.index]
                self.adata=sc.AnnData(counts)
                self.adata.X = csr_matrix(self.adata.X, dtype=np.float32)
                positions=pd.read_csv(os.path.join(self.path,'position.tsv'),sep='\t')
                self.adata.obsm['spatial'] = np.array(positions[['y','x']])
                self.adata.write(Path(self.path)/ 'raw.h5ad')
            else:
                self.adata = sc.read_h5ad(os.path.join(self.path, 'raw.h5ad'))
            if self.platform=='Slide_seqV2':
                self.adata.obsm['spatial'][:,1]=-1*self.adata.obsm['spatial'][:,1]
        else:
            self.adata = sc.read_h5ad(self.path+'.h5ad')
        self.adata.var_names_make_unique()
        if self.used_barcode:
            used_barcode=np.load(self.path+'/used_barcode.npy')
            self.adata=self.adata[used_barcode]

    def preprocess(self):
        print('Preprocess!')
        sc.pp.filter_genes(self.adata, min_cells=self.min_cells)
        sc.pp.filter_genes(self.adata, min_counts=self.min_counts)
        filter_specialgenes(self.adata)
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.fdim)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.scale(self.adata, zero_center=False, max_value=10)


    def generate_gene_expression(self):
        print('Generate gene expression!')
        adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        self.adata=adata_Vars
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        self.adata.obsm['gene_feat'] = feat

    def load_label(self):
        print('Load label!')

        if 'Region' in self.adata.obs.columns:
            self.adata.obs['ground_truth'] = self.adata.obs['Region']
        elif 'Annotation' in self.adata.obs.columns:
            self.adata.obs['ground_truth'] = self.adata.obs['Annotation']
        elif 'cluster' in self.adata.obs.columns:
            self.adata.obs['ground_truth'] = self.adata.obs['cluster']
        else:
            self.adata.obs['ground_truth'] = self.adata.obs['leiden']

        self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]



    def run(self):
        self.load_data()
        self.preprocess()
        self.generate_gene_expression()
        if self.label:
            self.load_label()
        self.adata.uns['smooth_r'] = self.smooth_r
        self.adata=gaussian_smooth_data(self.adata,self.d)
        self.adata=construct_adjacency_matrix(self.adata,self.num_neighbors,self.num_pruning,self.d)
        self.adata.write('./results/'+self.dataset+'/'+self.slice+'/data_processing.h5ad')
        print('adata load done!')

        return self.adata


class LoadBatchAdata:
    def __init__(self, path,catalogue, dataset, slice_list,config,args,path3='',join='inner',adata=None,used_barcode=False):
        self.path = path
        self.catalogue = catalogue
        self.dataset = dataset
        self.slice_list = slice_list
        self.label = args.label
        self.filter_na = args.label
        self.platform=args.platform
        self.config = config
        self.args = args
        self.path3 = path3
        self.used_barcode = used_barcode
        self.join=join
        self.fdim = config['fdim']
        self.smooth_r = config['smooth_r']
        self.min_cells=config['min_cells']
        self.min_counts=config['min_counts']
        self.num_neighbors=config['num_neighbors']
        self.num_pruning=config['num_pruning']
        self.adata = adata

    def load_data(self):
        print('Load batch data!')
        batch_adata=[]
        spatial_dict={}
        index_x=0
        index_y=0
        space=0
        for i,slice in enumerate(self.slice_list):
            print('Load '+slice+' data!')
            path=os.path.join(self.path, slice)
            if self.platform=='10x':
                Load=Load10xAdata(path,self.dataset,slice,self.config,self.args)
                Load.load_data()
                if self.label:
                    Load.load_label()
                data=Load.adata
                data.obsm['spatial'][:, 1]=-1*data.obsm['spatial'][:,1]
                self.d=Load.d
                space=10*self.smooth_r*self.d
            else:
                if self.args.parameter=='integration_cross_platform':
                    self.args.platform=slice
                Load=LoadAdata(path,self.dataset,slice,self.config,self.args,used_barcode=self.used_barcode)
                Load.load_data()
                if self.label:
                    Load.load_label()
                data=Load.adata
                space=10*self.smooth_r
            if i==0:
                index_x=data.obsm['spatial'][:,0].min()
                index_y=data.obsm['spatial'][:,1].min()
            adjust_y=data.obsm['spatial'][:,1].min()-index_y
            data.obsm['spatial'][:, 1]=data.obsm['spatial'][:, 1]-adjust_y

            x=data.obsm['spatial'][:,0]
            x=x-x.min()
            span=x.max()
            data.obsm['spatial'][:,0]=x+index_x
            index_x=index_x+span
            if self.args.parameter!='integration_horizontal':
                index_x=index_x+space
            data.obs['slice']=slice
            batch_adata.append(data)
        self.adata = sc.concat(batch_adata, join=self.join, label='slice', keys=self.slice_list, index_unique='-')
        self.adata.X=np.nan_to_num(self.adata.X,nan=0.0)
        self.adata.uns['spatial']=spatial_dict


    def preprocess(self):
        print('Preprocess!')
        sc.pp.filter_genes(self.adata, min_cells=self.min_cells)
        sc.pp.filter_genes(self.adata, min_counts=self.min_counts)
        filter_specialgenes(self.adata)
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.fdim)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.scale(self.adata, zero_center=False, max_value=10)


    def generate_gene_expression(self):
        print('Generate gene expression!')
        adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        self.adata=adata_Vars
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        self.adata.obsm['gene_feat'] = feat

    def run(self):
        self.load_data()
        self.preprocess()
        self.generate_gene_expression()
        if self.platform=='10x':
            self.adata.uns['smooth_r'] = self.smooth_r*self.d
        else:
            self.adata.uns['smooth_r'] = self.smooth_r
        self.adata=gaussian_smooth_data(self.adata)

        adj_list=[]
        for slice in self.slice_list:
            print('Construct  '+slice+' adjacency matrix!')
            data=self.adata[self.adata.obs['slice']==slice]
            adj=construct_adjacency_matrix(data,self.num_neighbors,self.num_pruning).obsm['graph_neigh']
            adj_list.append(adj)
        self.adata.obsm['graph_neigh']=block_diag(*adj_list)
        # adj=construct_adjacency_matrix(self.adata,self.num_neighbors,self.num_pruning).obsm['graph_neigh']
        # self.adata.obsm['graph_neigh']=adj
        self.adata.write('./results/'+self.catalogue+'/'+self.dataset+self.path3+'/data_processing.h5ad')
        print('adata load done!')
        return self.adata