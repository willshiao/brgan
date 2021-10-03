import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from dsloader.util import kron_graph, random_binary, make_fractional


class KroneckerDataset (Dataset):

    def __init__(self, kron_iter=4, seed_size=4, fixed_seed=None, num_graphs=1, perms_per_graph=256, progress_bar=False):
        self.kron_iter = kron_iter
        self.seed_size = seed_size


        self.num_nodes = seed_size ** (kron_iter + 1)
        self.seeds = []
        self.matrices = []

        num_iter = range(num_graphs)
        if progress_bar:
            from tqdm import tqdm
            num_iter = tqdm(num_iter)

        for i in num_iter:
            seed = random_binary(seed_size, use_sparsity=False)
            self.seeds.append(seed)
            if fixed_seed is not None:
                k_g = kron_graph(fixed_seed, n=kron_iter).astype(np.float)
            else:
                k_g = kron_graph(seed, n=kron_iter).astype(np.float)
            for j in range(perms_per_graph):
                self.matrices.append(make_fractional(k_g, inplace=False))
        
        
    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        return torch.tensor(self.matrices[idx])
