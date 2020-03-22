import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

import numpy as np

from layers import GraphConvolution
from torch.nn.parameter import Parameter



class GCNModelSIGVAE(nn.Module):
    def __init__(self, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, gdc='ip', ndist = 'Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(GCNModelSIGVAE, self).__init__()

        self.gce = GraphConvolution(ndim, hidden_dim1, dropout, act=F.relu)
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        if ndist == 'Bernoulli':
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':
            self.ndist == tdist.Normal(
                    torch.tensor([0.], device=self.device),
                    torch.tensor([1.], device=self.device))
        elif ndist == 'Exponential':
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        # K and J are defined in http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # Algorthm 1.
        self.K = copyK
        self.J = copyJ
        self.ndim = ndim


    def encode(self, x, adj):
        assert len(x.shape) == 3, 'The input shape dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.
        e = self.ndist.sample(torch.Size([self.K+self.J, x.shape[1], self.ndim]))
        e = torch.squeeze(e, -1)
        hiddene = self.gce(e, adj)
        hiddenx = self.gc1(x, adj)
        hidden1 = hiddenx + hiddene
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu), eps

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        emb_mu = mu[self.K:, :]
        emb_logvar = logvar[self.K:, :]

        z, eps = self.reparameterize(emb_mu, emb_logvar)
        return self.dc(z), mu, logvar, z, eps


class GraphDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, zdim, dropout, gdc='ip'):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = Parameter(torch.FloatTensor(torch.Size([1, zdim])))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rk_lgt)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        assert self.zdim == z.shape[2], 'zdim not compatible!'

        # The variable 'rk' in the code is the square root of the same notation in
        # http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # i.e., instead of do Z*diag(rk)*Z', we perform [Z*diag(rk)] * [Z*diag(rk)]'.
        rk_lgt = torch.clamp(self.rk_lgt, min=-np.Inf, max=0)
        rk = torch.sigmoid(rk_lgt)

        # Z shape: [J, N, zdim]
        # Z' shape: [J, zdim, N]
        z = z.mul(rk.view(1, 1, self.zdim))
        adj_lgt = torch.bmm(z, torch.transpose(z, 1, 2))

        if self.gdc == 'ip':
            adj = torch.sigmoid(adj_lgt)
        elif self.gdc == 'bp':
            # 1 - exp( - exp(ZZ'))
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())


        if self.training:
            adj_lgt = - torch.log(1 / (adj + self.SMALL) - 1 + self.SMALL)
            return adj_lgt
        else:
            adj_mean = torch.mean(adj, dim=0, keepdim=True)
            adj_mean_lgt = - torch.log(1 / (adj_mean + self.SMALL) - 1 + self.SMALL)
            return adj_mean_lgt





# class GCNModelVAE(nn.Module):
#     def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
#         super(GCNModelVAE, self).__init__()
#         self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
#         self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
#         self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
#         self.dc = GraphDecoder(dropout, act=lambda x: x)

#     def encode(self, x, adj):
#         hidden1 = self.gc1(x, adj)
#         return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def forward(self, x, adj):
#         mu, logvar = self.encode(x, adj)
#         mu = torch.mean(mu, dim=0, keepdim=False)
#         logvar = torch.mean(logvar, dim=0, keepdim=False)
#         z = self.reparameterize(mu, logvar)
#         return self.dc(z), mu, logvar


# class GraphDecoder(nn.Module):
#     """Decoder for using inner product for prediction."""

#     def __init__(self, dropout, act=torch.sigmoid):
#         super(GraphDecoder, self).__init__()
#         self.dropout = dropout
#         self.act = act


#     def forward(self, z):
#         z = F.dropout(z, self.dropout, training=self.training)
#         adj = self.act(torch.mm(z, z.t()))
#         return adj