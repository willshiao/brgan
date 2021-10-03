'''Defines residual GCN discriminators'''

import torch
import torch.nn as nn
from ggan.layers import GraphConvolution

class FixedResidualGcnDiscriminator(nn.Module):
    def __init__(self, num_nodes=500, dropout=0.5):
        super(FixedResidualGcnDiscriminator, self).__init__()
        self.num_nodes = num_nodes

        self.gcn1 = GraphConvolution(self.num_nodes, self.num_nodes)
        self.dropout = nn.Dropout()
        self.leaky1 = nn.LeakyReLU(inplace=True)

        self.gcn2 = GraphConvolution(self.num_nodes, self.num_nodes)
        self.leaky2 = nn.LeakyReLU(inplace=True)

        self.gcn3 = GraphConvolution(self.num_nodes, self.num_nodes)
        self.leaky3 = nn.LeakyReLU(inplace=True)

        self.gcn4 = GraphConvolution(self.num_nodes, self.num_nodes)
        self.leaky4 = nn.LeakyReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.final_layer = nn.Linear(self.num_nodes ** 2, 1)

    def forward(self, adj):
        I = torch.eye(self.num_nodes).cuda()
        x1 = self.gcn1(I, adj)
        # print(f'Post-GCN size: {x.shape}')
        x1 = self.leaky1(self.dropout(x1))
        x2 = self.leaky2(self.gcn2(x1, adj))
        x3 = self.leaky3(self.gcn3(x1, adj))
        x4 = self.leaky4(self.gcn4(x1, adj))
        # print(f'x-size: {x4.shape}')

        mats = [x1, x2, x3, x4]
        stacked = torch.stack(mats, dim=1) \
            .view(adj.size(0), len(mats), self.num_nodes, self.num_nodes)
        # print(f'stacked shape: {stacked.shape}')
        pooled = nn.functional.max_pool3d(stacked, kernel_size=(4, 1, 1))
        # print(f'post-pool shape: {pooled.shape}')
        pooled = self.sig(pooled)

        # print(f'Almost-final shape: {x.shape}')
        x = pooled.view(pooled.size(0), -1)
        # print(f'Reshaped: {x.shape}')

        return self.final_layer(x), x

class RepairedResidualGcnDiscriminator(nn.Module):
    def __init__(self, num_nodes=500, dropout=0.5):
        super(RepairedResidualGcnDiscriminator, self).__init__()
        self.num_nodes = num_nodes

        self.gcn1 = GraphConvolution(self.num_nodes, self.num_nodes)
        self.dropout = nn.Dropout()
        self.leaky1 = nn.LeakyReLU(inplace=True)

        self.gcn2 = GraphConvolution(self.num_nodes, self.num_nodes)
        self.leaky2 = nn.LeakyReLU(inplace=True)

        self.gcn3 = GraphConvolution(self.num_nodes, self.num_nodes)
        self.leaky3 = nn.LeakyReLU(inplace=True)

        self.gcn4 = GraphConvolution(self.num_nodes, self.num_nodes)
        self.leaky4 = nn.LeakyReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.final_layer = nn.Linear(self.num_nodes ** 2, 1)

    def forward(self, adj):
        I = torch.eye(self.num_nodes).cuda()
        x1 = self.gcn1(I, adj)
        # print(f'Post-GCN size: {x.shape}')
        x1 = self.leaky1(self.dropout(x1))
        x2 = self.leaky2(self.gcn2(x1, adj))
        x3 = self.leaky3(self.gcn3(x2, adj))
        x4 = self.leaky4(self.gcn4(x3, adj))
        # print(f'x-size: {x4.shape}')

        mats = [x1, x2, x3, x4]
        stacked = torch.stack(mats, dim=1) \
            .view(adj.size(0), len(mats), self.num_nodes, self.num_nodes)
        # print(f'stacked shape: {stacked.shape}')
        pooled = nn.functional.max_pool3d(stacked, kernel_size=(4, 1, 1))
        # print(f'post-pool shape: {pooled.shape}')
        pooled = self.sig(pooled)

        # print(f'Almost-final shape: {x.shape}')
        x = pooled.view(pooled.size(0), -1)
        # print(f'Reshaped: {x.shape}')

        return self.final_layer(x), x
