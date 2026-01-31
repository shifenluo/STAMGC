import seaborn as sns
from scipy import stats
from typing import List, Optional, Union
import gseapy as gp

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import scanpy as sc
import json

from sklearn.decomposition import PCA



def filter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).upper().startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).upper().startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)

def knn_adj_matrix(coor,num_neighbor,metric='euclidean'):
    n_spot=coor.shape[0]
    adj = np.zeros([n_spot, n_spot], dtype=float)
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbor + 1, metric=metric).fit(coor)
    distances, indices = nbrs.kneighbors(coor, n_neighbors=num_neighbor + 1, return_distance=True)
    for i in range(n_spot):
        adj[i, indices[i]] = 1
    return adj


def gaussian_smooth_data(adata,d=2):
    print('Gaussian smooth data!')
    if d==2:
        coor = adata.obsm['spatial']
    else:
        adata.obsm['xyz']=np.column_stack((adata.obs['x'], adata.obs['y'], adata.obs['z']))
        coor=adata.obsm['xyz']
    gene = adata.obsm['gene_feat']
    sigma=adata.uns['smooth_r']

    dist_matrix = cdist(coor, coor)

    weights = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    weights /= np.sum(weights,1,keepdims=True)  # normalization

    smooth_gene = weights @ gene
    adata.obsm['smooth_gene'] = smooth_gene
    return adata

def construct_adjacency_matrix(adata,num_neighbors,num_pruning,d=2):
    print('Construct adjacency matrix!')
    if d==2:
        coor = pd.DataFrame(adata.obsm['spatial'])
    else:
        # adata.obsm['xyz']=np.column_stack((adata.obs['x'], adata.obs['y'], adata.obs['z']))
        coor=pd.DataFrame(adata.obsm['xyz'])
    coor.index = adata.obs.index
    coor_k_interaction = knn_adj_matrix(coor,num_neighbors,metric='euclidean')

    coor_k_adj = coor_k_interaction + coor_k_interaction.T
    coor_k_adj = np.where(coor_k_adj > 1, 1, coor_k_adj)
    coor_k_adj=coor_k_adj-np.eye(coor_k_adj.shape[0])

    adata.obsm['graph_neigh_coor'] = coor_k_adj

    smooth_gene = pd.DataFrame(adata.obsm['smooth_gene'])
    smooth_gene_cosine=cdist(smooth_gene,smooth_gene,'cosine')

    for i in range(num_pruning):
        smooth_gene_cosine_matrix=coor_k_interaction*smooth_gene_cosine
        del_col_idx = smooth_gene_cosine_matrix.argmax(axis=1)  # The column index of the first maximum value in each row
        coor_k_interaction[np.arange(len(del_col_idx)), del_col_idx] = 0

    coor_adj = coor_k_interaction + coor_k_interaction.T
    coor_adj = np.where(coor_adj > 1, 1, coor_adj)
    coor_adj=coor_adj-np.eye(coor_adj.shape[0])

    adata.obsm['graph_neigh'] = coor_adj
    adata.obsm['negative_mask'] = coor_k_adj-coor_adj

    return adata

def draw_spatial_domain(adata,key,title,file_name,platform,batch_size,spot_size,path1,path2,path3,cmap=None):
    if platform == '10x' and batch_size==1:
        sc.pl.spatial(adata, img_key="hires", color=key, title=title, cmap=cmap,
                          show=False)
        plt.savefig('./results/' + path1 + '/' + path2+path3 + '/'+file_name+'.png', bbox_inches='tight',
                        dpi=600)
        plt.show()
    else:
        sc.pl.embedding(adata, basis='spatial', color=key, title=title, cmap=cmap,size=spot_size,
                            show=False)
        plt.savefig('./results/' + path1 + '/' + path2+path3 + '/'+file_name+'.png', bbox_inches='tight',
                        dpi=600)
        plt.show()

def read_json(json_path):
    # Read and parse JSON files
    with open(json_path, 'r') as file:
        data = json.load(file)
    number = data['fiducial_diameter_fullres']
    integer_part = int(number)# Extract the integer part

    return integer_part


def adj_to_edge_index(adj):
    row, col = torch.where(adj != 0)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def edge_index_to_adj(edge_index,n):
    adj=torch.zeros((n,n)).to(edge_index.device)
    row, col = edge_index
    adj[row,col]=1
    return adj


def volcano_plot(adata, target_cluster: str, target_cluster1: str=None, cluster_key: str = 'cluster',
                 method: str = 'wilcoxon', top_n: int = 10,
                 figsize: tuple = (10, 8),
                 logfc_threshold: float = 0.25,
                 pvalue_threshold: float = 0.05,
                 xlim: Optional[tuple] = (-3, 3),
                 save_path: Optional[str] = None):
    """
    Draw a volcano plot to show the differentially expressed genes between the target cluster and other clusters

    Parameters:
    - adata: AnnData object
    - target_cluster: Target cluster label
    - cluster_key: Column name of cluster label in adata.obs
    - method: Differential expression analysis method ('wilcoxon', 't-test', 't-test_overestim_var')
    - top_n: Mark the top N most significant genes
    - figsize: Image size
    - logfc_threshold: Log fold change threshold
    - pvalue_threshold: P-value threshold
    - save_path: Save path
    """
    if target_cluster1 is None:
        print(f"正在进行差异表达分析: {target_cluster} vs 其他聚类...")

        adata_copy = adata.copy()
        title=f'Volcano Plot: {target_cluster} vs Others\nTop {top_n} genes highlighted'
    else:
        print(f"正在进行差异表达分析: {target_cluster} vs {target_cluster1}...")
        adata_copy = adata[adata.obs[cluster_key].isin([target_cluster,target_cluster1])].copy()
        title=f'Volcano Plot: {target_cluster} vs {target_cluster1}\nTop {top_n} genes highlighted'

    adata_copy.obs['de_group'] = (adata_copy.obs[cluster_key] == target_cluster).astype(str)

    # Perform differential expression analysis
    sc.tl.rank_genes_groups(adata_copy, 'de_group', method=method,
                            groups=['True'], reference='False')

    # extraction results
    result = sc.get.rank_genes_groups_df(adata_copy, group='True')
    result = result.dropna(subset=['pvals_adj', 'logfoldchanges'])


    result['neg_log10_padj'] = -np.log10(result['pvals_adj'])


    result['significant'] = ((result['neg_log10_padj'] > -np.log10(pvalue_threshold)) &
                             (np.abs(result['logfoldchanges']) > logfc_threshold))


    top_genes = result.nlargest(top_n, 'neg_log10_padj')


    plt.figure(figsize=figsize)
    in_range = result[(result['logfoldchanges'] >= xlim[0]) & (result['logfoldchanges'] <= xlim[1])]

    plt.scatter(in_range[~in_range['significant']]['logfoldchanges'],
                in_range[~in_range['significant']]['neg_log10_padj'],
                alpha=0.5, s=20, color='gray', label='Not significant')

    up_genes = in_range[in_range['significant'] & (in_range['logfoldchanges'] > 0)]
    plt.scatter(up_genes['logfoldchanges'], up_genes['neg_log10_padj'],
                alpha=0.7, s=20, color='red', label='Up-regulated')

    down_genes = in_range[in_range['significant'] & (in_range['logfoldchanges'] < 0)]
    plt.scatter(down_genes['logfoldchanges'], down_genes['neg_log10_padj'],
                alpha=0.7, s=20, color='blue', label='Down-regulated')


    for _, gene_info in top_genes.iterrows():
        x, y = gene_info['logfoldchanges'], gene_info['neg_log10_padj']

        if x < xlim[0]: x = xlim[0]
        if x > xlim[1]: x = xlim[1]
        plt.annotate(gene_info['names'],
                     (x, y),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.axhline(y=-np.log10(pvalue_threshold), color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=logfc_threshold, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=-logfc_threshold, color='black', linestyle='--', alpha=0.5)

    plt.xlabel('Log Fold Change')
    plt.ylabel('-Log10 Adjusted P-value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return top_genes,result



def plot_spatial_gene_expression(adata, gene: str,
                                 spatial_key: str = 'spatial',
                                 figsize: tuple = (5, 4),
                                 cmap: str = 'viridis',
                                 spot_size: float = 15,
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None,
                                 platform: str = '10x'):
    """
    Display the expression heatmap of a single gene across the entire space

    Parameters:
    - adata: AnnData object, containing spatial coordinate information
    - gene: The gene name to be displayed
    - spatial_key: The key name of spatial coordinates in adata.obsm
    - figsize: image size
    - cmap: color mapping ('viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdYlBu_r', 'RdBu_r')
    - spot_size: size of the spot
    - title: Custom Title
    - save_path: Save path
    """

    if gene not in adata.var_names:
        print(f"错误: 基因 '{gene}' 不存在于数据中")
        print(f"可用的基因示例: {list(adata.var_names[:10])}")
        return


    if spatial_key not in adata.obsm:
        print(f"错误: 空间坐标 '{spatial_key}' 不存在于 adata.obsm 中")
        return

    print(f"绘制基因 '{gene}' 的空间表达分布...")


    spatial_coords = adata.obsm[spatial_key]
    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]


    expression = adata[:, gene].X.toarray().flatten()


    plt.figure(figsize=figsize)


    scatter = plt.scatter(x_coords, y_coords, c=expression,
                          cmap=cmap, s=spot_size, alpha=0.8)


    if title is None:
        title = gene
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    plt.gca().set_aspect('equal')

    if platform == '10x':
        plt.gca().invert_yaxis()

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Expression Level', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


def violin_plot_comparison(adata, genes: Union[str, List[str]],
                           target_cluster: str,target_cluster1: str=None, cluster_key: str = 'cluster',
                           figsize: tuple = (8, 6),
                           save_path: Optional[str] = None):
    """
    Draw a violin plot comparison of the specified gene in the target cluster and other clusters

    Parameters:
    - adata: AnnData object
    - genes: The gene names or gene list to be plotted
    - target_cluster: Target cluster label
    - cluster_key: The column name of the clustering label in adata.obs
    - figsize: image size
    - save_path: Save path
    """
    if isinstance(genes, str):
        genes = [genes]
    if target_cluster1 is None:
        print(f"绘制小提琴图: {genes} 在 {target_cluster} vs 其他聚类")

        adata_copy = adata.copy()
        adata_copy.obs['comparison_group'] = adata_copy.obs[cluster_key].apply(
            lambda x: target_cluster if x == target_cluster else 'Others'
        )
    else:
        print(f"绘制小提琴图: {genes} 在 {target_cluster} vs {target_cluster1}")
        adata_copy = adata[adata.obs[cluster_key].isin([target_cluster,target_cluster1])].copy()
        adata_copy.obs['comparison_group'] = adata_copy.obs[cluster_key].apply(
            lambda x: target_cluster if x == target_cluster else target_cluster1
        )


    plot_order = [target_cluster, 'Others']
    plot_palette = {target_cluster: 'lightcoral', 'Others': 'lightblue'}

    n_genes = len(genes)
    fig, axes = plt.subplots(1, n_genes, figsize=(figsize[0] * n_genes, figsize[1]))

    if n_genes == 1:
        axes = [axes]

    for i, gene in enumerate(genes):
        if gene not in adata_copy.var_names:
            print(f"警告: 基因 {gene} 不存在于数据中")
            axes[i].text(0.5, 0.5, f'Gene {gene}\nnot found',
                         ha='center', va='center', transform=axes[i].transAxes)
            continue

        expression_data = pd.DataFrame({
            'Expression': adata_copy[:, gene].X.toarray().flatten(),
            'Group': adata_copy.obs['comparison_group'],
            'Cluster': adata_copy.obs[cluster_key]
        })

        # Draw a violin diagram
        sns.violinplot(data=expression_data, x='Group', y='Expression',
                       ax=axes[i], order=plot_order, palette=plot_palette)

        # Add scatter plot
        # sns.stripplot(data=expression_data, x='Group', y='Expression',
        #              ax=axes[i], color='black', alpha=0.3, size=3, jitter=True)

        target_expr = expression_data[expression_data['Group'] == target_cluster]['Expression']
        other_expr = expression_data[expression_data['Group'] == 'Others']['Expression']

        if len(target_expr) > 1 and len(other_expr) > 1:
            stat, pval = stats.ttest_ind(target_expr, other_expr)
            axes[i].set_title(f'{gene}\np-value: {pval:.2e}')
        else:
            axes[i].set_title(f'{gene}')

        axes[i].set_ylabel('Expression Level')
        axes[i].set_xlabel('')

        axes[i].axhline(y=target_expr.mean(), color='red', linestyle='--', alpha=0.7)
        axes[i].axhline(y=other_expr.mean(), color='blue', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


import re


def go_analysis(gene_list: List[str],
                background_genes: List[str],
                organism: str = "Human",
                outdir: str = "go_analysis_results",
                top_n: int = 10,
                figsize: tuple = (12, 16)) -> pd.DataFrame:
    """
    Run three types of GO ontology analysis and combine the results into a single bar chart

    Parameters:
    - gene_list: The list of genes you are interested in
    - background_genes: List of background genes
    - organism: species, "Human", "Mouse", "Yeast", etc
    - gene_id_type: gene ID type, "symbol", "entrez", "ensembl"
    - outdir: the folder where the results are saved
    - top_n: Display the top N most significant results for each GO ontology
    - figsize: image size

    Return:
    - A merged DataFrame containing all GO ontology results
    """
    print(f"开始运行GO分析并合并结果...")
    print(f"输入基因数量: {len(gene_list)}")
    print(f"背景基因数量: {len(background_genes)}")

    ontologies = ['Biological_Process', 'Molecular_Function', 'Cellular_Component']
    ontology_colors = {'Biological_Process': '#1f77b4', 'Molecular_Function': '#ff7f0e',
                       'Cellular_Component': '#2ca02c'}  # Blue, orange, green

    all_top_results = []

    for ont in ontologies:
        print(f"\n--- 正在分析 GO {ont} ---")

        current_outdir = f"{outdir}/GO_{ont}"

        try:
            enr = gp.enrichr(gene_list=gene_list,
                             gene_sets=f'GO_{ont}_2023',
                             organism=organism,
                             outdir=current_outdir,
                             no_plot=True,
                             cutoff=0.05)

            results_df = enr.results

            if results_df.empty:
                print(f"GO {ont}: 未找到显著富集的条目。")
                continue

            top_results = results_df.sort_values('Adjusted P-value', ascending=True).head(top_n).copy()

            top_results['Ontology'] = ont
            top_results['Color'] = ontology_colors[ont]

            all_top_results.append(top_results)
            print(f"GO {ont}: 找到 {len(results_df)} 个显著条目，已选取前 {top_n} 个。")

        except Exception as e:
            print(f"GO {ont} 分析失败: {e}")

    # Combine all results
    if not all_top_results:
        print("所有GO本体均未找到显著结果，无法绘图。")
        return pd.DataFrame()

    combined_results = pd.concat(all_top_results, ignore_index=True)


    combined_results['sort_key'] = combined_results['Ontology'].map(
        {'Biological_Process': 0, 'Molecular_Function': 1, 'Cellular_Component': 2})
    combined_results = combined_results.sort_values(['sort_key', 'Adjusted P-value'], ascending=[True, True])

    combined_results['Clean_Term'] = combined_results['Term'].apply(lambda x: re.sub(r'\s*\(GO:\d+\)', '', x))

    combined_results['Adjusted P-value'].replace(0, 1e-300, inplace=True)

    combined_results['-log10(Adjusted P-value)'] = -np.log10(combined_results['Adjusted P-value'])

    # --- Start drawing ---
    plt.figure(figsize=figsize)

    bar_colors = combined_results['Color'].tolist()

    barplot = sns.barplot(x='-log10(Adjusted P-value)', y='Clean_Term', data=combined_results, palette=bar_colors)

    bp_end = len(combined_results[combined_results['Ontology'] == ontologies[0]]) - 1
    mf_end = bp_end + len(combined_results[combined_results['Ontology'] == ontologies[1]])

    if bp_end >= 0:
        plt.axhline(y=bp_end + 0.5, color='gray', linestyle='--', linewidth=1.5)

    if mf_end > bp_end:
        plt.axhline(y=mf_end + 0.5, color='gray', linestyle='--', linewidth=1.5)

    legend_handles = []
    for ont, color in ontology_colors.items():

        if ont in combined_results['Ontology'].values:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=ont))

    if legend_handles:
        plt.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(1.0, 0.0),
                   frameon=True, fancybox=True, shadow=True)

    plt.title(f'GO Enrichment Analysis (Top {top_n} per Ontology)', fontsize=18, pad=20)
    plt.xlabel('-log10(Adjusted P-value)', fontsize=14)
    plt.ylabel('')
    plt.tight_layout()

    save_path = f"{outdir}/Combined_GO_Enrichment.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"\n合并结果图已保存到: {save_path}")

    plt.show()

    combined_results.to_csv(f"{outdir}/Combined_GO_Results.csv", index=False)
    print(f"合并结果表已保存到: {outdir}/Combined_GO_Results.csv")

    return combined_results

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2025):
    np.random.seed(random_seed)
    # import os
    # os.environ['R_LIBS_USER'] = "/home/lsfcj/.conda/envs/STAMGC/lib/R/library"
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    # print('res:',res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    # adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, key='emb', method='mclust',labels='domain'):
    n_components = 20
    if adata.obsm[key].shape[1] < n_components:
        n_components=adata.obsm[key].shape[1]
    pca = PCA(n_components=n_components, random_state=2025)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata=adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs[labels] = adata.obs['mclust']
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=2025).fit(embedding)
        kmeans_result = [i + 1 for i in kmeans.labels_]
        adata.obs[labels] = np.array(kmeans_result).astype('int')
    if labels != 'pseudo_labels':
        adata.obs[labels] =adata.obs[labels].astype('category')


def generate_pseudo_labels(smooth_gene, n_clusters=7):
    n_components = 20
    if smooth_gene.shape[1] < n_components:
        n_components=smooth_gene.shape[1]
    pca = PCA(n_components=n_components, random_state=2025)
    embedding = pca.fit_transform(smooth_gene.copy())

    kmeans = KMeans(n_clusters=n_clusters, random_state=2025).fit(embedding)
    pseudo_labels = np.array([i + 1 for i in kmeans.labels_])
    return pseudo_labels

if __name__ == "__main__":
    # 假设 data 是一个 n x m 的矩阵
    data = np.array([
        1,4,7,1,4,7,2,1,2
    ])

    # 假设 adjmatrix 是一个 n x n 的邻接矩阵
    adjmatrix = torch.tensor([
        [1,0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    matrix = torch.tensor([
        [0,1, 1],
        [0, 1, 1],
        [1, 0, 1]
    ])
    print(((adjmatrix-matrix)==1).float())