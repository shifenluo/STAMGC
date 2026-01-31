import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from package.loss import mse_loss
from package.utils import adj_to_edge_index, clustering,  generate_pseudo_labels, draw_spatial_domain


# ---------- multi-branch MLP fusion module ----------
class MultiHeadMLP(nn.Module):
    """
    head_num independent MLPs, concatenated along the last dimension,
    with WO dimensionality reduction
    """
    def __init__(self, in_dim, head_dim=32, head_num=4, out_dim=64, dropout=0.1):
        super(MultiHeadMLP,self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.d_h = head_dim * head_num          # Total dimension after splicing
        self.out_dim = out_dim or in_dim        # If not reducing dimensions, keep in_dim

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, head_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_dim, head_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(head_num)
        ])

        self.WO = nn.Sequential(
            nn.Linear(self.d_h, self.out_dim),
            nn.ReLU(inplace=True)
        ) if self.d_h != self.out_dim else nn.Identity()

    def forward(self, x):
        # x: [N, in_dim]
        outs = [h(x) for h in self.heads]          # List[Tensor[N, head_dim]]
        concat = torch.cat(outs, dim=-1)           # [N, H*head_dim]
        return self.WO(concat)                     # [N, out_dim]


# ---------- STAMGC ----------
class model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                  head_num=4, head_dim=32, dropout=0.1):
        super(model,self).__init__()
        # 1. GNN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.decoder = GCNConv(out_channels, in_channels)

        # 2. multi-branch MLP fusion module
        self.mhp = MultiHeadMLP(out_channels,
                                head_dim=head_dim,
                                head_num=head_num,
                                out_dim=out_channels,
                                dropout=dropout)

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        z = F.relu(self.conv2(x, edge_index))

        mhp=self.mhp(z)
        h=F.relu(self.decoder(z, edge_index))


        return z,mhp,h



class STAMGC(nn.Module):
    def __init__(self,adata,config,args,path3='',cluster_method='kmeans',load=False):
        super(STAMGC,self).__init__()
        self.load=load
        self.nfeat=config['fdim']
        self.nhid=config['nhid']
        self.out=config['out']
        self.head_num=config['head_num']
        self.head_dim=config['head_dim']
        self.adata=adata
        self.batch_size=args.batch_size
        self.path1=args.catalogue if args.batch_size>1 else args.dataset
        self.path2=args.dataset if args.batch_size>1 else args.slice
        self.path3=path3
        self.cluster_method=cluster_method
        self.num_clusters = config['num_clusters']
        self.label = args.label
        self.platform=args.platform
        self.spot_size=args.spot_size
        self.args=args
        self.config=config
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gama=config['gama']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.epoches = config['epoches']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.features = torch.FloatTensor(self.adata.obsm['gene_feat'].copy()).to(self.device)
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device)

        self.model=model(self.nfeat,self.nhid,self.out,head_num=self.head_num,head_dim=self.head_dim,dropout=self.dropout).to(self.device)#GNNDGIAutoEncoder
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def generate_pseudo_labels(self,load=False):
        print('=== Generate pseudo labels! ===')
        if load:
            self.adata.obs['pseudo_labels'] =np.load('./results/' + self.path1 + '/' + self.path2+self.path3 + '/pseudo_labels.npy')
        else:
            # clustering(self.adata,self.num_clusters,'smooth_gene',self.cluster_method,'pseudo_labels')
            self.adata.obs['pseudo_labels']=generate_pseudo_labels(self.adata.obsm['smooth_gene'],self.num_clusters)
            np.save('./results/' + self.path1 + '/' + self.path2 +self.path3+ '/pseudo_labels.npy',self.adata.obs['pseudo_labels'])
        draw_spatial_domain(self.adata,'pseudo_labels','pseudo labels','pseudo_labels',self.platform,self.batch_size,self.spot_size,self.path1,self.path2,self.path3,cmap='magma')



    def train(self):
        features=self.features
        graph_neigh=self.graph_neigh
        pseudo_labels=torch.FloatTensor(self.adata.obs['pseudo_labels']).to(self.device)
        edge_index=adj_to_edge_index(graph_neigh)

        T_LOSS=[]
        b_xent = nn.BCEWithLogitsLoss()
        print('=== train ===')
        for epoch in range(self.epoches):
            self.model.train()
            self.optimizer.zero_grad()

            z,mhp,h=self.model(features,edge_index)

            reconstitution_loss=mse_loss(h,features)

            contrast_loss1=self.contrastive_loss1(z, graph_neigh,pseudo_labels)

            sim=self.sim(mhp,mhp)
            # sim=self.sim(z,z)
            contrast_loss2=b_xent(sim,graph_neigh)

            total_loss = self.alpha * reconstitution_loss+self.beta * contrast_loss1+self.gama * contrast_loss2
            total_loss.backward()
            self.optimizer.step()


            T_LOSS.append(total_loss.item())
            print("epoch:",epoch,
                " total_loss:",total_loss.item(),
                ' alpha:',self.alpha,' beta:',self.beta,' gama:',self.gama)
        torch.save(self.model.state_dict(), './results/'+self.path1+'/'+self.path2+self.path3+'/model.pt')


        x = range(self.epoches)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, T_LOSS)
        ax.set_title('T_LOSS')

        plt.tight_layout()
        plt.show()


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def nei_con_loss(self, z, adj,pseudo_labels):#,negative_mask,z2
        '''neighbor contrastive loss'''

        f = lambda x: torch.exp(x)#
        intra_view_sim = f(self.sim(z, z))

        pseudo_labels_negative_mask = (pseudo_labels.view(-1, 1) != pseudo_labels.view(1, -1)).float()

        intra_negative_view = intra_view_sim * pseudo_labels_negative_mask

        loss = ((intra_view_sim.mul(adj)).sum(1)) / (
                intra_negative_view.sum(1))

        return -torch.log(loss)


    def contrastive_loss1(self, z, adj,pseudo_labels):
        ret = self.nei_con_loss(z, adj,pseudo_labels)
        ret = ret.mean()

        return ret


    def eva(self):
        print("=== load ===")
        self.model.load_state_dict(torch.load('./results/'+self.path1+'/'+self.path2+self.path3+'/model.pt'))
        self.model.eval()
        features=self.features
        graph_neigh=self.graph_neigh
        edge_index = adj_to_edge_index(graph_neigh)
        z,mlp,h=self.model(features, edge_index)
        self.adata.obsm['z'] = z.detach().cpu().numpy()
        self.adata.obsm['mlp'] = mlp.detach().cpu().numpy()
        self.adata.obsm['h'] = h.detach().cpu().numpy()
        print('embedding generated, go clustering')

    def cluster(self):
        clustering(self.adata, self.num_clusters, key='z', method=self.cluster_method,labels='domain')

        if self.label:
            print('calculate metric ARI')
            ARI = metrics.adjusted_rand_score(self.adata.obs['domain'], self.adata.obs['ground_truth'])
            self.adata.uns['ari'] = ARI
            NMI = metrics.normalized_mutual_info_score(self.adata.obs['domain'], self.adata.obs['ground_truth'])
            self.adata.uns['nmi'] = NMI
            print('ARI:', ARI)
            print('NMI:', NMI)
            title='STAMGC (ARI={:.2f},NMI={:.2f})'.format(ARI,NMI)

            draw_spatial_domain(self.adata, 'ground_truth', 'Manual annotation (' + self.path2 + ')', 'ground_truth', self.platform,
                            self.batch_size, self.spot_size, self.path1, self.path2,self.path3, cmap=None)

        else:
            print("calculate SC and DB")
            SC = silhouette_score(self.adata.obsm['z'], self.adata.obs['domain'])
            DB = davies_bouldin_score(self.adata.obsm['z'], self.adata.obs['domain'])
            self.adata.uns['sc'] = SC
            self.adata.uns['db'] = DB
            print('SC:', SC)
            print('DB:', DB)
            title='STAMGC (SC={:.2f},DB={:.2f})'.format(SC,DB)
        draw_spatial_domain(self.adata, 'domain', title, 'STAMGC', self.platform,
                            self.batch_size, self.spot_size, self.path1, self.path2,self.path3, cmap=None)


    def run(self):
        self.generate_pseudo_labels(load=self.load)
        self.train()
        self.eva()
        self.cluster()

        self.adata.write('./results/'+self.path1+'/'+self.path2+self.path3+'/STAMGC.h5ad')
        print('complete!')

        return self.adata