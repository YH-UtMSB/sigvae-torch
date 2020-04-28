from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import GCNModelSIGVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--model', type=str, default='gcn_vae', help="models used.")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--monit', type=int, default=50, help='Number of epochs to train before a test')
parser.add_argument('--edim', type=int, default=32, help='Number of units in noise epsilon.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--encsto', type=str, default='semi', help='encoder stochasticity.')
parser.add_argument('--gdc', type=str, default='ip', help='type of graph decoder')
parser.add_argument('--noise-dist', type=str, default='Bernoulli',
                    help='Distriubtion of random noise in generating psi.')
parser.add_argument('--K', type=int, default=15,
                    help='number of samples to draw for MC estimation of h(psi).')
parser.add_argument('--J', type=int, default=20,
                    help='Number of samples to draw for MC estimation of log-likelihood.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    # Set tensor dtype to float16
    # torch.set_default_tensor_type(torch.HalfTensor)

    adj, features = load_data(args.dataset_str)
    _, n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)  

    model = GCNModelSIGVAE(
        args.edim, feat_dim, args.hidden1, args.hidden2, args.dropout,
        encsto=args.encsto, 
        gdc=args.gdc, 
        ndist=args.noise_dist,
        copyK=args.K, 
        copyJ = args.J, 
        device=args.device
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None

    model.to(args.device)
    features = features.to(args.device)
    adj_norm = adj_norm.to(args.device)
    adj_label = adj_label.to(args.device)
    pos_weight = pos_weight.to(args.device)


    

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar, z, z_scaled, eps, rk = model(features, adj_norm)
        loss_rec, loss_prior, loss_post = loss_function(
            preds=recovered, 
            labels=adj_label,
            mu=mu, 
            logvar=logvar, 
            emb=z, 
            eps=eps, 
            n_nodes=n_nodes,
            norm=norm, 
            pos_weight=pos_weight
        )

        WU = np.min([epoch/300., 1.])
        reg = (loss_post - loss_prior) * WU / (n_nodes**2)
        
        loss_train = loss_rec + WU * reg
        # loss_train = loss_rec
        loss_train.backward()

        cur_loss = loss_train.item()
        cur_rec = loss_rec.item()
        # cur_rec_bce = loss_rec1.item()
        optimizer.step()

        hidden_emb = z_scaled.detach().cpu().numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, val_edges, val_edges_false, args.gdc)
        

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "rec_loss=", "{:.5f}".format(cur_rec),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )
        # print(rk.detach().cpu().numpy())


        if((epoch+1) % args.monit == 0):
            model.eval()
            recovered, mu, logvar, z, z_scaled, eps, rk = model(features, adj_norm)
            hidden_emb = z_scaled.detach().cpu().numpy()
            roc_score, ap_score = get_roc_score(hidden_emb, test_edges, test_edges_false, args.gdc)
            rslt = "Test ROC score: {:.4f}, Test AP score: {:.4f}\n".format(roc_score, ap_score)
            print("\n", rslt, "\n")
            # with open("results.txt", "a+") as f:
            #     f.write(rslt)

    print("Optimization Finished!")
    


if __name__ == '__main__':
    gae_for(args)
