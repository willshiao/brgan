import torch.nn as nn

class FCDiscriminator(nn.Module):
    def __init__(self, num_nodes=500, dropout=0.5, layer_size=128):
        super(FCDiscriminator, self).__init__()
        self.num_nodes = num_nodes
        self.layer_size = layer_size
        self.layers = nn.Sequential(
            nn.Linear(self.num_nodes, self.layer_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),

            nn.Linear(self.layer_size, self.layer_size * 2),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),

            nn.Linear(self.layer_size * 2, self.layer_size),
            nn.Sigmoid()
        )
        self.final_layer = nn.Linear(self.layer_size, 1)

    def forward(self, adj):
        x = self.layers(adj)
        return self.final_layer(x), x
