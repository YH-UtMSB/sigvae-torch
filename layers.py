import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    GCN layer, based on https://arxiv.org/abs/1609.02907
    that allows MIMO
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        """
        if the input features are a matrix -- excute regular GCN,
        if the input features are of shape [K, N, Z] -- excute MIMO GCN with shared weights.
        """
        # An alternative to derive XW (line 32 to 35)
        # W = self.weight.view(
        #         [1, self.in_features, self.out_features]
        #         ).expand([input.shape[0], -1, -1])
        # support = torch.bmm(input, W)

        support = torch.stack(
                [torch.mm(inp, self.weight) for inp in torch.unbind(input, dim=0)],
                dim=0)
        output = torch.stack(
                [torch.spmm(adj, sup) for sup in torch.unbind(support, dim=0)],
                dim=0)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
