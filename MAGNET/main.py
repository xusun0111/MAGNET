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
import sys
from torch_geometric.utils import remove_self_loops
from model import FDE_encoder, GraphTransformer_encoder, GCN_encoder, MAGNET
from utils import dimensional_sample_random, DAD_edge_index, freeze_test, get_embedding
from torch_geometric.utils import degree

torch.cuda.empty_cache()

def run(args):
    with open(args.log_dir, 'a') as f:
        f.write('\n\n\n')
        f.write(str(args))
    free_gpu_id = args.GPU_ID
    torch.cuda.set_device(int(free_gpu_id))
    dataset = args.dataset
    data_dir = args.datadir
    nb_epochs = args.nb_epochs
    lr = args.lr
    wd = args.wd
    hid_units = args.hid_units
    num_hop = args.num_hop
    activator = nn.PReLU if args.activator == 'PReLU' else nn.ReLU
    torch_geometric.seed.seed_everything(args.seed)
    seed = args.seed
    sample_size = args.sample_size
    feature_signal_dim = args.feature_signal_dim
    alpha = args.alpha
    beta = args.beta
    gama = args.gama
    if_rand = True if args.if_rand=='True' else False

    data = load_dataset(dataset, data_dir)[0]
    edge_index = data.edge_index.cuda()
    second_order_edge_index = compute_second_order_neighbors(edge_index)
    second_order_edge_index = second_order_edge_index.cuda()
    data.second_order_edge_index = second_order_edge_index

    dnn = FDE_encoder(sample_size, feature_signal_dim*2, feature_signal_dim, activator)
    gcn = GCN_encoder(feature_signal_dim, hid_units, activator)
    trans = GraphTransformer_encoder(feature_signal_dim, hid_units, activator)
    model = MAGNET(D_NN=dnn, GCN=gcn, Trans=trans, S_mtd=dimensional_sample_random, sample_size=sample_size)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
    if torch.cuda.is_available():
        data = data.cuda()
        model = model.cuda()
    model.update_sample(data.x, data.second_order_edge_index, if_rand=if_rand)
    with tqdm(total=nb_epochs, desc='(T)') as pbar:
        for epoch in range(nb_epochs):
            model.train()
            optimizer.zero_grad()
            z_g, z_t = model(data.x, data.second_order_edge_index)
            loss_neg = model.loss_neg(z_g)
            loss_pos = model.loss_pos(z_t, data.second_order_edge_index)
            loss_fde = model.loss_fde()
            loss = alpha * loss_fde + beta * loss_pos + gama * loss_neg
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({  'loss_fde': loss_fde.item(),
                                'loss_pos': loss_pos.item(),
                                'loss_neg': loss_neg.item()   })

            pbar.update()
                        
    model.eval()

    z = get_embedding(data.x, data.second_order_edge_index, model, num_hop, if_rand=if_rand)

    # m, r  = freeze_test(z, data.y, train_ratio=0.1, test_ratio=0.8, test_num=20)
    m, r  = freeze_test(z, data.y, train_ratio=0.6, test_ratio=0.2, test_num=20)
    # m, r  = freeze_test(z, data.y.long(), train_ratio=0.6, test_ratio=0.2, test_num=20)
    # m, r  = freeze_test(z, data.y.long(), train_ratio=0.5, test_ratio=0.25, test_num=20)
    with open(args.log_dir, 'a') as f:
        f.write('\n')
        f.write(' mean: '+str(m)+' std: '+ str(r))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #setting arguments
    parser = argparse.ArgumentParser('MAGNET')
    parser.add_argument('--dataset', type=str, default='Wisconsin',
    help="""Dataset name: Cora, CiteSeer, PubMed, dblp, Photo, Computers, CS, Physics,
    ogbn-products, ogbn-arxiv, Wiki, ppi, Cornell, Texas, Wisconsin,
    chameleon, crocodile, squirrel, actor, roman_empire, amazon_ratings,
    minesweeper, tolokers, questions, chameleon_filtered, squirrel_filtered""")
    parser.add_argument('--datadir', type=str, default='../../../datasets/', help='./data/dir/')
    parser.add_argument('--log_dir', type=str, default='./log/logCora.txt', help='./log/dir/')
    parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID')
    parser.add_argument('--seed', type=int, default=777, help='seed')

    parser.add_argument('--nb_epochs', type=int, default=300, help='Number of epochs') #training epochs
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--activator', type=str, default='PReLU', help='Activator name: PReLU, ReLU')
    parser.add_argument('--if_rand', type=str, default='False', help='feature sample if_rand: True, False')
    
    parser.add_argument('--hid_units', type=int, default=1024, help='representation size')
    parser.add_argument('--sample_size', type=int, default=183, help='node sample size')
    parser.add_argument('--feature_signal_dim', type=int, default=1024, help='feature signal dim')
    parser.add_argument('--alpha', type=float, default=1, help='hyper-parameter of loss_fde')
    parser.add_argument('--beta', type=float, default=100, help='hyper-parameter of loss_pos')
    parser.add_argument('--gama', type=float, default=0.3, help='hyper-parameter of loss_neg')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    run(args)
