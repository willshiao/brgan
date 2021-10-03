'''Bilinear GAN generator classes'''
import torch
import torch.nn as nn

# Best bilinear model so far
class BetterSharedFFBilinearGenerator(nn.Module):
    def __init__(self, latent_dim=100, layer_size=128, num_nodes=500, rank=30, extra_dim=False):
        super(BetterSharedFFBilinearGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.latent_dim = latent_dim
        self.extra_dim = extra_dim

        shared_layers = [
            nn.Linear(latent_dim, layer_size),
            nn.Linear(layer_size, layer_size * 2),
            nn.BatchNorm1d(layer_size * 2),
            nn.ReLU(inplace=True),
            # New block
            nn.Linear(layer_size * 2, layer_size * 4),
            nn.BatchNorm1d(layer_size * 4),
        ]

        output_layers = [
            [
                nn.Linear(layer_size * 4, layer_size * 2),
                nn.Linear(layer_size * 2, num_nodes * rank),
                nn.Hardtanh(min_val=0)
                # nn.BatchNorm1d(layer_size * 4),
            ] for _ in range(2)
        ]

        self.shared = nn.Sequential(*shared_layers)
        self.output1 = nn.Sequential(*output_layers[0])
        self.output2 = nn.Sequential(*output_layers[1])
        self.output_factors = False
    
    def set_factor_output(self, new_val):
        self.output_factors = new_val
        return True

    def forward(self, noise):
        batch_sz = noise.shape[0]
        S = self.shared(noise)
        A = self.output1(S).view(batch_sz, self.num_nodes, self.rank)
        B = self.output2(S).view(batch_sz, self.rank, self.num_nodes)
        res = torch.bmm(A, B)

        if self.extra_dim:
            out = res.view(batch_sz, 1, self.num_nodes, self.num_nodes)
        elif not self.output_factors:
            out = res.view(batch_sz, self.num_nodes, self.num_nodes)
        if self.output_factors:
            return (out, (A, B))
        else:
            return out

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class BaselineGenerator(nn.Module):
    def __init__(self, latent_dim=100, layer_size=128, num_nodes=500, rank=30, extra_dim=False):
        super(BaselineGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.latent_dim = latent_dim
        self.extra_dim = extra_dim

        shared_layers = [
            nn.Linear(latent_dim, layer_size),
            nn.Linear(layer_size, layer_size * 2),
            nn.BatchNorm1d(layer_size * 2),
            nn.ReLU(inplace=True),
            # New block
            nn.Linear(layer_size * 2, layer_size * 4),
            nn.BatchNorm1d(layer_size * 4),
        ]

        output_layer = [
            nn.Linear(layer_size * 4, layer_size * 2),
            nn.Linear(layer_size * 2, num_nodes * num_nodes),
            nn.Hardtanh(min_val=0)
        ]

        self.shared = nn.Sequential(*shared_layers)
        self.output = nn.Sequential(*output_layer)
        self.output_factors = False
    
    def set_factor_output(self, new_val):
        self.output_factors = new_val
        return True

    def forward(self, noise):
        batch_sz = noise.shape[0]
        S = self.shared(noise)
        res = self.output(S)

        if self.extra_dim:
            out = res.view(batch_sz, 1, self.num_nodes, self.num_nodes)
        elif not self.output_factors:
            out = res.view(batch_sz, self.num_nodes, self.num_nodes)
        if self.output_factors:
            return (out, (None, None))
        else:
            return out

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
