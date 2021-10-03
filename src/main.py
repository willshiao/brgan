#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wandb
from training import Trainer
from torch.utils.data.sampler import RandomSampler
from ggan import ModelZoo
from torch.utils.data import Dataset
from dsloader.graphrnn_creator import create
import networkx as nx
import json
import numpy as np
import pickle
import random
import torch
import torch.utils.data as utils
import sys

from dsloader import util
from os import path
from pathlib import Path
from tqdm import tqdm


# In[2]:


random.seed(199231318)
np.random.seed(681212354)

cuda = True if torch.cuda.is_available() else False
print('CUDA is enabled' if cuda else 'CUDA is not enabled')

# VERSION = '1.6.0'
VERSION = '1.1.0'
ALLOW_OVERRIDE = False

opt = {
    'n_epochs': 5000,  # number of epochs of training
    'batch_size': 64,  # size of the batches
    'n_permutations': 5020,  # number of permutations of the graph
    'gen_lr': 0.001,  # adam: learning rate
    'disc_lr': 0.001,  # learning rate for discriminator
    # (b1, b2): decay of first order momentum of gradient & first order momentum of gradient
    'betas': (0.5, 0.999),
    'n_cpu': 32,  # number of cpu threads to use during batch generation
    'latent_dim': 100,  # dimensionality of the latent space
    'rank': 20,  # rank used for bilinear size
    'gen_layer_size': 128,  # size of layers in generator
    'disc_layer_size': 1024,  # size of layers in discriminator
    'sample_interval': 800,  # interval between graph sampling
    'save_interval': 250,  # interval between model saving
    'print_interval': 100,  # interval between printing loss info
    'plot_interval': 200,  # interval between plotting
    'loss_interval': 10,  # interval between loss sampling
    'model_name': 'CT_Residual_BL_Citeseer_RMS',
    'slice_size': 50,  # graph slice size
    'fully_random': False,
    'comment': '',
    # generator class name, used by ModelZoo
    'gen_class': 'BetterSharedFfBilinearGenerator',
    # discriminator class name, used by ModelZoo
    'disc_class': 'FixedResidualGcnDiscriminator',
    'dataset': 'citeseer_small',
    'rank_lambda': 0.01,
    'penalty_type': 'fro',
    'n_graph_sample_batches': 10,
    'rank_penalty_method': 'A', # A, B, C, D, or E
    'eval_every': 100
}

for arg in sys.argv[1:]:
    pieces = arg.lstrip('-').split('=')
    if len(pieces) != 2:
        print(f'Invalid argument format: {arg}')
        sys.exit(1)
    l, r = pieces
    setting_ver = False

    # Get type of existing argument
    if l.lower() == 'version':
        VERSION = r
        setting_ver = True
        print(f'Setting version to {VERSION}')
    elif isinstance(opt[l], int):
        # convert to int
        opt[l] = int(r)
    elif isinstance(opt[l], str):
        opt[l] = r
    elif isinstance(opt[l], bool):
        opt[l] = bool(r)
    elif isinstance(opt[l], float):
        opt[l] = float(r)
    else:
        print('Unknown opt type for {l}')
        sys.exit(2)

    if not setting_ver:
        print(f'Setting opt[{l}] to {opt[l]}')


BASE_PATH = 'results/'
PATH_PREFIX = path.join(BASE_PATH, '{}_v{}'.format(opt['model_name'], VERSION))
SAVE_PATH = path.join(PATH_PREFIX, 'models/')
IMG_PATH = path.join(PATH_PREFIX, 'images/')
STATS_PATH = path.join(PATH_PREFIX, 'stats/')


# In[7]:


if path.exists(PATH_PREFIX):
    if ALLOW_OVERRIDE:
        print(
            f'WARNING: ERROR: path ({PATH_PREFIX}) already exists, but ALLOW_OVERRIDE is set')
    else:
        raise Exception(
            f'ERROR: path ({PATH_PREFIX}) already exists, no files created')

to_make = [SAVE_PATH, IMG_PATH, STATS_PATH]
for p in to_make:
    Path(p).mkdir(parents=True, exist_ok=True)

print('Saving params')
with open(path.join(PATH_PREFIX, 'params.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4, separators=(',', ': '))
wandb.init(config=opt)

gs = create({'graph_type': opt['dataset']})
print(f'Using {len(gs)} graphs')


# In[3]:


everything = []
for g in tqdm(gs):
    everything.extend(util.get_bfs_orderings(g))


# In[4]:


cube = torch.zeros((1, opt['slice_size'], opt['slice_size'], len(everything)))

for i in range(len(everything)):
    ten = torch.tensor(nx.to_numpy_array(everything[i]))
    dim = ten.shape[0]
    cube[0, :dim, :dim, i] = ten
print(everything[-1].size())


# In[5]:


class BasicDataset (Dataset):
    def __init__(self, data):
        self.cube = data

    def __len__(self):
        return self.cube.shape[-1]

    def __getitem__(self, idx):
        return (self.cube[:, :, :, idx], 0)


# ## Start main stuff

# In[6]:


print('Saving input matrices')
with open(path.join(PATH_PREFIX, 'input_mats.pkl'), 'wb') as f:
    pickle.dump(everything, f)


# In[8]:


zoo = ModelZoo()


# In[9]:


# Loss functions
adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

print(f"Using {opt['gen_class']} for generator")
print(f"Using {opt['disc_class']} for discriminator")

# Initialize generator and discriminator
gen_class = zoo.get_model(opt['gen_class'])
disc_class = zoo.get_model(opt['disc_class'])
discriminator = disc_class(num_nodes=opt['slice_size'])
generator = gen_class(
    num_nodes=opt['slice_size'], layer_size=opt['gen_layer_size'], rank=opt['rank'], extra_dim=True)

discriminator.float()
generator.float()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()


# In[10]:


# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt['gen_lr'])
optimizer_D = torch.optim.RMSprop(
    discriminator.parameters(), lr=opt['disc_lr'])

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# In[11]:


test_dataset = BasicDataset(cube)
sampler = RandomSampler(test_dataset)
data_loader = utils.DataLoader(
    test_dataset, batch_size=opt['batch_size'], sampler=sampler)


# In[12]:


trainer = Trainer(generator, discriminator, optimizer_G, optimizer_D, use_cuda=cuda, epoch_print_every=3,
                  print_every=150, checkpoint_path=SAVE_PATH, checkpoint_every=opt['save_interval'], rank_lambda=opt['rank_lambda'],
                  penalty_type=opt['penalty_type'], wandb=wandb, rank_penalty_method=opt['rank_penalty_method'],
                  n_graph_sample_batches=opt['n_graph_sample_batches'], batch_size=opt['batch_size'], eval_every=opt['eval_every'],
                  gs=gs)


# In[13]:


try:
    trainer.train(data_loader, opt['n_epochs'], save_training_gif=True)
except KeyboardInterrupt:
    print('Caught keyboard interrupt, saving model...')
    torch.save(generator.state_dict(), path.join(
        SAVE_PATH, '{}_generator_v{}-final'.format(opt['model_name'], VERSION)))
    torch.save(discriminator.state_dict(), path.join(
        SAVE_PATH, '{}_discriminator_v{}-final'.format(opt['model_name'], VERSION)))
    torch.save(optimizer_G.state_dict(), path.join(
        SAVE_PATH, '{}_optimizerG_v{}-final'.format(opt['model_name'], VERSION)))
    torch.save(optimizer_D.state_dict(), path.join(
        SAVE_PATH, '{}_optimizerD_v{}-final'.format(opt['model_name'], VERSION)))
    print('Done, bye!')

print('Done, saving model...')
torch.save(generator.state_dict(), path.join(
    SAVE_PATH, '{}_generator_v{}-final'.format(opt['model_name'], VERSION)))
torch.save(discriminator.state_dict(), path.join(
    SAVE_PATH, '{}_discriminator_v{}-final'.format(opt['model_name'], VERSION)))
torch.save(optimizer_G.state_dict(), path.join(
    SAVE_PATH, '{}_optimizerG_v{}-final'.format(opt['model_name'], VERSION)))
torch.save(optimizer_D.state_dict(), path.join(
    SAVE_PATH, '{}_optimizerD_v{}-final'.format(opt['model_name'], VERSION)))
