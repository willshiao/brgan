import torch
import torch.nn as nn

class ModdedSharedSvdGenerator(nn.Module):
    def __init__(self, latent_dim=100, layer_size=128, num_nodes=500, rank=30, extra_dim=False):
        super(ModdedSharedSvdGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.latent_dim = latent_dim
        self.extra_dim = extra_dim
        self.output_factors = False

        shared_layers = [
            nn.Linear(latent_dim, layer_size),
            nn.Linear(layer_size, layer_size * 2),
            nn.BatchNorm1d(layer_size * 2),
            nn.ReLU(inplace=True),
            # New block
            nn.Linear(layer_size * 2, layer_size * 4),
            nn.BatchNorm1d(layer_size * 4),
        ]

        mat_output_layers = [
            [
                nn.Linear(layer_size * 4, num_nodes * rank)
            ] for _ in range(2)
        ]
        sigma_output_layers = [
            nn.Linear(layer_size * 4, rank)
        ]

        self.shared = nn.Sequential(*shared_layers)
        self.output1 = nn.Sequential(*mat_output_layers[0])
        self.output2 = nn.Sequential(*mat_output_layers[1])
        self.output_sigma = nn.Sequential(*sigma_output_layers)

    def set_factor_output(self, new_val):
        self.output_factors = new_val
        return True

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

    def forward(self, noise):
        batch_sz = noise.shape[0]
        S = self.shared(noise)
        U = self.output1(S).view(batch_sz, self.num_nodes, self.rank)
        Vt = self.output2(S).view(batch_sz, self.rank, self.num_nodes)
        sig = self.output_sigma(S).view(batch_sz, self.rank)
        sig_diag = torch.diag_embed(sig)
        U_scaled = torch.bmm(U, sig_diag)
        res = torch.bmm(U_scaled, Vt)

        if self.extra_dim:
            out = res.view(batch_sz, 1, self.num_nodes, self.num_nodes)
        elif not self.output_factors:
            out = res.view(batch_sz, self.num_nodes, self.num_nodes)

        if self.output_factors:
            return (out, (U, Vt))
        else:
            return out

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
