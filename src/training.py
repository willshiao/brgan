import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import graphrnn as grnn
from os import path
from dsloader import util
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import imageio


class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10.0, ct_weight=2.0, M=0.1, critic_iterations=5, print_every=50, epoch_print_every=5,
                 use_cuda=False, checkpoints=True, checkpoint_every=500, checkpoint_path='models/',
                 file_name='ctgan', rank_lambda=0.0, penalty_type='fro', wandb=None, rank_penalty_method='A',
                 n_graph_sample_batches=10, batch_size=64, eval_every=250, gs=None, eval_method='graph'):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'CT': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.ct_weight = ct_weight
        self.M = M
        self.batch_size = batch_size
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.epoch_print_every = epoch_print_every
        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.file_name = file_name
        self.rank_lambda = rank_lambda
        self.penalty_type = penalty_type
        self.rank_penalty_method = rank_penalty_method
        if rank_penalty_method not in ('A', 'B', 'C', 'D', 'E'):
            raise Exception('Invalid rank penalty method')
        if gs is None:
            raise Exception('gs argument required')
        self.gs = gs
        self.wandb = wandb
        self.to_log = {}
        self.n_graph_sample_batches = n_graph_sample_batches
        self.eval_every = eval_every
        self.eval_method = eval_method

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _remove_unconnected(self, G):
        to_remove = []
        for node in G.nodes():
            if len(G.edges(node)) == 0:
                to_remove.append(node)

        G.remove_nodes_from(to_remove)
        return G

    def wandb_log(self, data, **kwargs):
        if self.wandb is not None:
            self.wandb.log(data, **kwargs)

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)[0]
        d_generated = self.D(generated_data)[0]

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data) 
        self.losses['GP'].append(float(gradient_penalty))

        # Get consistency term
#         print(f'Data shape: {data.size()}')
        consistency_term = self._consistency_term(data)
        self.losses['CT'].append(float(consistency_term))

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty + consistency_term
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.to_log['discriminator_loss'] = d_loss
        self.to_log['consistency_term'] = float(consistency_term)
        self.to_log['gradient_penalty'] = float(gradient_penalty)
        self.losses['D'].append(float(d_loss))

    def _get_rank_penalty(self, M, factor=None):
        if self.rank_penalty_method == 'A':
            if self.penalty_type == 'fro':
                return self.rank_lambda * torch.norm(M @ torch.transpose(M, 1, 2), p=self.penalty_type)
            else:
                whole = M @ torch.transpose(M, 1, 2)
                total = 0.0
                for i in range(whole.size(0)):
                    total += torch.norm(whole[i, :, :], p=self.penalty_type)
                return self.rank_lambda * total
        elif self.rank_penalty_method == 'B':
            # Same as A
            return self.rank_lambda * torch.norm(M @ torch.transpose(M, 1, 2), p=self.penalty_type)
        elif self.rank_penalty_method == 'C':
            return self.rank_lambda * torch.norm(torch.eye(M.size(1), device='cuda') - M @ torch.transpose(M, 1, 2), p=self.penalty_type)
        elif self.rank_penalty_method == 'D':
            if factor == 'A':
                tmp = torch.transpose(M, 1, 2) @ M
                eye = torch.eye(M.size(2), device='cuda')
            elif factor == 'B':
                tmp = M @ torch.transpose(M, 1, 2)
                eye = torch.eye(M.size(1), device='cuda')
            else:
                print('No factor specified for penalty method D')
                return None
            return self.rank_lambda * torch.norm(tmp - eye, p=self.penalty_type)
        elif self.rank_penalty_method == 'E':
            if factor == 'A':
                tmp = torch.transpose(M, 1, 2) @ M
            elif factor == 'B':
                tmp = M @ torch.transpose(M, 1, 2)
            else:
                print('No factor specified for penalty method E')
                return None
            diag = torch.diag_embed(torch.diagonal(tmp, dim1=1, dim2=2))
            return self.rank_lambda * torch.norm(tmp - diag, p=self.penalty_type)
        else:
            raise Exception('Unknown rank penalty type!')

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        if self.rank_penalty_method == 'A':
            generated_data = self.sample_generator(batch_size)
        else:
            self.G.set_factor_output(True)
            generated_data, factors = self.sample_generator(batch_size)
            self.G.set_factor_output(False)

        # Get additional penalty
        #print(f'Shape: {generated_data.shape}')

        # Calculate loss and optimize
        d_generated = self.D(generated_data)[0]
        g_loss = - d_generated.mean()
        g_loss_original = float(g_loss)
        if self.rank_lambda > 0:
            graph_size = data.size()[2]
            if self.rank_penalty_method == 'A':
                A = generated_data.view(batch_size, graph_size, graph_size)
                rank_penalty = self._get_rank_penalty(A)
            else:
                rank_penalty = self._get_rank_penalty(factors[0], factor='A') + self._get_rank_penalty(factors[1], factor='B')
            rank_penalty_original = float(rank_penalty)
            g_loss += rank_penalty
            self.to_log['rank_penalty_loss'] = rank_penalty_original
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.to_log['generator_loss_original'] = g_loss_original
        self.to_log['generator_loss'] = float(g_loss)
        self.losses['G'].append(float(g_loss))
        
    def _consistency_term(self, real_data):
        d1, d_1 = self.D(real_data)
        d2, d_2 = self.D(real_data)

        consistency_term = (d1 - d2).norm(2, dim=1) + 0.1 * \
            (d_1 - d_2).norm(2, dim=1) - self.M
        return consistency_term.mean()

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)[0]

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(float(gradients.norm(2, dim=1).mean()))

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, print_vals=False, epoch_num=-1):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data[0])
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data[0])

            if print_vals and i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("CT: {}".format(self.losses['CT'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))
        self.to_log['epoch'] = epoch_num
        if self.eval_every > 0 and (epoch_num+1) % self.eval_every == 0:
            res = self.evaluate_gen()
            if res is not None:
                (deg_stat, clustering_stat, orbit_stat, valid_graphs, rank_mean, rank_std, rank_med) = res
                self.to_log['degree_mmd'] = deg_stat
                self.to_log['clustering_mmd'] = clustering_stat
                self.to_log['orbit_mmd'] = orbit_stat
                self.to_log['mean_mmd'] = np.mean(res[:3])
                self.to_log['mmd_graphs'] = valid_graphs
                self.to_log['rank_mean'] = rank_mean
                self.to_log['rank_std'] = rank_std
                self.to_log['rank_med'] = rank_med
        self.wandb_log(self.to_log)


    def save_checkpoint(self, epoch):
        torch.save(self.G.state_dict(), path.join(self.checkpoint_path,
                                                  '{}_generator_epoch-{}.model'.format(self.file_name, epoch)))
        torch.save(self.D.state_dict(), path.join(self.checkpoint_path,
                                                  '{}_discriminator_epoch-{}.model'.format(self.file_name, epoch)))
        torch.save(self.G_opt.state_dict(), path.join(self.checkpoint_path,
                                                      '{}_optimizerG_epoch-{}.model'.format(self.file_name, epoch)))
        torch.save(self.D_opt.state_dict(), path.join(self.checkpoint_path,
                                                      '{}_optimizerD_epoch-{}.model'.format(self.file_name, epoch)))

    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(self.batch_size))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            should_print = (epoch % self.epoch_print_every == 0)
            if should_print:
                print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader, print_vals=should_print, epoch_num=epoch)

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)

            if (epoch+1) % self.checkpoint_every == 0 and self.checkpoints:
                print(f'========== Saving checkpoint at epoch #{epoch} ========')
                self.save_checkpoint(epoch)


        if save_training_gif:
            imageio.mimsave('{}/../training_{}_epochs.gif'.format(self.checkpoint_path, epochs),
                            training_progress_images)

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def evaluate_gen(self):
        if (self.eval_method == 'graph'):
            return self.eval_gen_graphs()
        else:
            print(f'ERROR: {self.eval_method} method not currently supported')
            return None
    
    def eval_gen_graphs(self):
        gen_G = []
        ranks = []
        for i in range(self.n_graph_sample_batches):
            z = self.G.sample_latent(self.batch_size)
            if self.use_cuda:
                z = z.cuda()
            res = self.G(z)
            g_np = res.detach().cpu().numpy()

            for j in range(g_np.shape[0]):
                (_, S, _) = np.linalg.svd(g_np[j, 0, :, :])
                ranks.append(util.approx_rank(S))

                tmp = g_np[j, 0, :, :].copy()
                util.graph_threshold(tmp, threshold=0.0001)
                graph = nx.from_numpy_array(tmp)
                self._remove_unconnected(graph)
                if graph.number_of_nodes() > 0:
                    gen_G.append(graph)
        print(f'Generated {len(gen_G)} graphs for evaluation')
        if len(gen_G) == 0:
            print('WARNING: no graphs were generated for evaluation')
            return None
        try:
            deg_stat = grnn.degree_stats(self.gs, gen_G)
            clustering_stat = grnn.clustering_stats(self.gs, gen_G)
            orbit_stat = grnn.orbit_stats_all(self.gs, gen_G)
        except Exception as e:
            print('WARNING: failed to calculate MMD scores:', e)
            return None
        return (deg_stat, clustering_stat, orbit_stat, len(gen_G), np.mean(ranks), np.std(ranks), np.median(ranks))
        

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]

