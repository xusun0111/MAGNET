
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor, WikiCS, ppi
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def load_dataset(dataset_name, dataset_dir):

    print('Dataloader: Loading Dataset', dataset_name)
    assert dataset_name in ['Cora', 'CiteSeer', 'PubMed',
                            'dblp', 'Photo' ,'Computers',
                            'CS' ,'Physics',
                            'ogbn-products', 'ogbn-arxiv', 'Wiki' ,'ppi',
                            'Cornell', 'Texas', 'Wisconsin',
                            'chameleon', 'crocodile', 'squirrel' ,'actor', 'roman_empire', 'amazon_ratings', 'minesweeper',
                            'tolokers', 'questions', 'chameleon_filtered', 'squirrel_filtered']

    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name,
                            transform=T.NormalizeFeatures())

    elif dataset_name == 'dblp':
        dataset = CitationFull(dataset_dir, name=dataset_name,
                               transform=T.NormalizeFeatures())

    elif dataset_name in ['Photo' ,'Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name,
                         transform=T.NormalizeFeatures())

    elif dataset_name in ['CS' ,'Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name,
                           transform=T.NormalizeFeatures())

    elif dataset_name in ['Wiki']:
        dataset = WikiCS(dataset_dir  )# ,
        # transform=T.NormalizeFeatures())
    elif dataset_name in ['ppi']:
        train = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'train')
        val = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'val')
        test = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'test')
        dataset = [train, val, test]

    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        return WebKB(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())

    elif dataset_name in ['chameleon', 'crocodile', 'squirrel']:
        use_geom_gcn = dataset_name in ['chameleon', 'squirrel']
        return WikipediaNetwork(
            dataset_dir,
            dataset_name,
            geom_gcn_preprocess=use_geom_gcn,
            transform=T.NormalizeFeatures())

    elif dataset_name == 'actor':
        dataset = Actor(root=dataset_dir, transform=T.NormalizeFeatures())

    elif dataset_name in ['roman_empire', 'amazon_ratings', 'minesweeper',
                          'tolokers', 'questions', 'chameleon_filtered', 'squirrel_filtered']:
        dataset_path = f'../hetgs/{dataset_name}.npz'
        data_npz = np.load(dataset_path)
        features = torch.tensor(data_npz['node_features'], dtype=torch.float)
        labels = torch.tensor(data_npz['node_labels'], dtype=torch.long)
        edge_index = torch.tensor(data_npz['edges'], dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        train_mask = torch.tensor(data_npz['train_masks'], dtype=torch.bool)
        val_mask = torch.tensor(data_npz['val_masks'], dtype=torch.bool)
        test_mask = torch.tensor(data_npz['test_masks'], dtype=torch.bool)
        data = Data(x=features, edge_index=edge_index, y=labels,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        dataset = [data]

    print('Dataloader: Loading success.')
    print(dataset[0])

    return dataset

def compute_second_order_neighbors(edge_index, top_k=5):
    device = edge_index.device
    num_nodes = int(edge_index.max().item() + 1)
    adj_dict = {}
    for u, v in edge_index.t().tolist():
        adj_dict.setdefault(u, set()).add(v)

    second_edge_weights = {}
    for u in adj_dict:
        for v in adj_dict[u]:
            if v in adj_dict:
                for w in adj_dict[v]:
                    if w != u and w not in adj_dict.get(u, set()):
                        second_edge_weights[(u, w)] = second_edge_weights.get((u, w), 0) + 1
    filtered_edges = []
    from collections import defaultdict
    candidate_edges = defaultdict(list)
    for (u, w), weight in second_edge_weights.items():
        candidate_edges[u].append((w, weight))

    for u, edges in candidate_edges.items():
        edges = sorted(edges, key=lambda x: x[1], reverse=True)[:top_k]
        for w, _ in edges:
            filtered_edges.append((u, w))

    if not filtered_edges:
        return torch.empty((2,0), device=device, dtype=torch.long)

    second_order_edge_index = torch.tensor(filtered_edges, device=device).t()
    second_order_edge_index = torch.cat([second_order_edge_index, second_order_edge_index[[1 ,0]]], dim=1)
    second_order_edge_index = torch.unique(second_order_edge_index, dim=1)
    print('filtered second_order_edge_index:', second_order_edge_index, second_order_edge_index.shape)
    return second_order_edge_index
