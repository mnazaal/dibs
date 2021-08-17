


import tqdm
import time

import jax.numpy as jnp
from jax import jit, vmap, random, grad, vjp, jvp, jacrev
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jax.lax import stop_gradient
from jax.ops import index, index_add, index_update, index_mul
from jax.nn import sigmoid, log_sigmoid
import jax.lax as lax
from jax.experimental import optimizers
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_multimap, tree_reduce

import numpy as onp

from dibs.utils.graph import acyclic_constr_nograd, elwise_acyclic_constr_nograd
from dibs.utils.func import (
    mask_topk, id2bit, bit2id, log_prob_ids, 
    particle_joint_empirical, particle_joint_mixture,
    expand_by,
)
from dibs.utils.tree import tree_unzip_leading

from dibs.eval.mmd import MaximumMeanDiscrepancy
from dibs.eval.metrics import neg_ave_log_likelihood, l1_edge_belief
from dibs.kernel.basic import StructuralHammingSquaredExponentialKernel
from dibs.exceptions import SVGDNaNError, SVGDNaNErrorLatent, SVGDNaNErrorTheta


class BatchedJointDotProductGraphSVGD:
    """
        n_vars:                     number of variables in graph
        n_dim:                      size of latent represention
        kernel :                    object satisfying `BasicKernel` signature with differentiable evaluation
        target_log_prior :          differentiable log prior probability using the probabilities of edges
        target_log_joint_prob :     differentiable or non-differentiable discrete log probability of target distribution
        optimizer:                  dictionary with at least keys `name` and `stepsize`
        alpha:                      inverse temperature parameter schedule of sigmoid; satisfies t -> float
        beta:                       inverse temperature parameter schedule of prior; satisfies t -> float
        gamma:                      scaling parameter schedule inside acyclicity constraint; satisfies t -> float
        tau:                        inverse temperature parameter schedule of gumbel-softmax; satisfies t -> float
        n_grad_mc_samples:          MC samples in gradient estimator
        n_grad_batch_size:          Minibatch size in taret log prob evaluation
        n_acyclicity_mc_samples:    MC samples in reparameterization of acyclicity constraint
        repulsion_in_prob_space:    whether to apply kernel to particles the probabilities they encode
        clip:                       lower and upper bound for particles across all dimensions
        grad_estimator_x:           'score' or 'reparam'
        grad_estimator_theta:       'hard' or 'reparam'
        score_function_baseline:    weight of addition in score function baseline; == 0.0 corresponds to not using a baseline
        latent_prior_std:           if == -1.0, latent variables have no prior other than acyclicity
                                    otherwise, uses Gaussian prior with mean = 0 and var = `latent_prior_std` ** 2
        constraint_prior_graph...   how acyclicity is defined in p(Z)
                                    - None:   uses matrix of probabilities and doesn't sample
                                    - 'soft': samples graphs using Gumbel-softmax in forward and backward pass
                                    - 'hard': samples graphs using Gumbel-softmax in backward but Bernoulli in forward pass
        graph_embedding_representation     if true, uses inner product embedding graph model
        fix_rotation:               if true, fixes latent representations U, V at row 0 to 1
    """

    def __init__(self, *, n_vars, n_dim, kernel, target_log_prior, target_log_joint_prob, target_log_joint_prob_no_batch, alpha, beta, gamma, tau, n_grad_mc_samples,
                 optimizer, n_acyclicity_mc_samples, grad_estimator_x='reparam', grad_estimator_theta='hard', score_function_baseline=0.0, clip=None,
                 repulsion_in_prob_space=False, latent_prior_std=-1.0, fix_rotation="not", constraint_prior_graph_sampling=None,
                 graph_embedding_representation=True, n_grad_batch_size=None, verbose=False):
        super(BatchedJointDotProductGraphSVGD, self).__init__()

        self.n_vars = n_vars
        self.n_dim = n_dim
        self.kernel = kernel
        self.target_log_prior = target_log_prior
        self.target_log_joint_prob = target_log_joint_prob
        self.target_log_joint_prob_no_batch = target_log_joint_prob_no_batch
        self.alpha = jit(alpha)
        self.beta = jit(beta)
        self.gamma = jit(gamma)
        self.tau = jit(tau)
        self.n_grad_mc_samples = n_grad_mc_samples
        self.n_grad_batch_size = n_grad_batch_size
        self.n_acyclicity_mc_samples = n_acyclicity_mc_samples
        self.repulsion_in_prob_space = repulsion_in_prob_space
        self.grad_estimator_x = grad_estimator_x
        self.grad_estimator_theta = grad_estimator_theta
        self.score_function_baseline = score_function_baseline
        self.latent_prior_std = latent_prior_std
        self.constraint_prior_graph_sampling = constraint_prior_graph_sampling
        self.graph_embedding_representation = graph_embedding_representation
        self.clip = clip
        self.fix_rotation = fix_rotation
        self.optimizer = optimizer

        self.verbose = verbose
        self.has_init_core_functions = False
        self.init_core_functions()

    def vec_to_mat(self, x):
        '''Reshapes particle to latent adjacency matrix form
            last dim gets shaped into matrix
            w:    [..., n_vars * n_vars]
            out:  [..., n_vars, n_vars]
        '''
        return x.reshape(*x.shape[:-1], self.n_vars, self.n_vars)

    def mat_to_vec(self, w):
        '''Reshapes latent adjacency matrix form to particle
            last two dims get flattened into vector
            w:    [..., n_vars, n_vars]
            out:  [..., n_vars * n_vars]
        '''
        return w.reshape(*w.shape[:-2], self.n_vars * self.n_vars)

    def particle_to_hard_g(self, x):
        '''Returns g corresponding to alpha = infinity for particles `x`
            x:   [..., n_vars, n_dim, 2]
            out: [..., n_vars, n_vars]
        '''
        if self.graph_embedding_representation:
            u, v = x[..., 0], x[..., 1]
            scores = jnp.einsum('...ik,...jk->...ij', u, v)
        else:
            scores = x

        g_samples = (scores > 0).astype(jnp.int32)

        # zero diagonal
        g_samples = index_mul(g_samples, index[..., jnp.arange(scores.shape[-1]), jnp.arange(scores.shape[-1])], 0)
        return g_samples

    def particle_to_sampled_log_prob(self, graph_ids, x, log_expectation_per_particle, t):
        '''
        Compute sampling-based SVGD log posterior for `graph_ids` using particles `x`
            
            graph_ids:  [n_graphs]     
            x:          [n_particles, n_vars, n_dim, 2]
            log_expectation_per_particle: [n_particles]
                        computed apriori from samples given the particles
            t:          [1, ]

            out:        [n_graphs]  log probabilities of `graph_ids`

        '''

        n_particles = x.shape[0]
        assert(self.n_vars <= 5)

        '''Compute sampling-based SVGD log posterior for provided graph ids'''
        g = id2bit(graph_ids, self.n_vars)

        # log p(D | G)
        # [batch_size, 1]
        g_log_probs = self.eltwise_log_prob(g)[..., jnp.newaxis]

        # log p(G | W)
        # [batch_size, n_particles]
        g_particle_likelihoods = self.double_eltwise_latent_log_prob(g, x, t)
    
        # [1, n_particles]
        denominator = log_expectation_per_particle[jnp.newaxis]

        # log p(G | D)
        svgd_sampling_log_posterior = logsumexp(g_log_probs + g_particle_likelihoods - denominator, axis=1) - jnp.log(n_particles)

        return svgd_sampling_log_posterior

        
    def init_core_functions(self):
        '''Defines functions needed for SVGD and uses jit'''

        def sig_(x, t):
            '''Sigmoid with parameter `alpha`'''
            return sigmoid(self.alpha(t) * x)
        self.sig = jit(sig_)

        def log_sig_(x, t):
            '''Log sigmoid with parameter `alpha`'''
            return log_sigmoid(self.alpha(t) * x)
        self.log_sig = jit(log_sig_)

        def edge_probs_(x, t):
            '''
            Edge probabilities encoded by latent representation
                x:      [..., d, k, 2]
                out:    [..., d, d]
            '''
            if self.graph_embedding_representation:
                u, v = x[..., 0], x[..., 1]
                scores = jnp.einsum('...ik,...jk->...ij', u, v)
            else:
                scores = x

            probs =  self.sig(scores, t)

            # mask diagonal since it is explicitly not modeled
            probs = index_mul(probs, index[..., jnp.arange(probs.shape[-1]), jnp.arange(probs.shape[-1])], 0.0)
            return probs

        self.edge_probs = jit(edge_probs_)

        def edge_log_probs_(x, t):
            '''
            Edge log probabilities encoded by latent representation
                x:      [..., d, k, 2]
                out:    [..., d, d], [..., d, d]
            The returned tuples are log(p) and log(1 - p)
            '''
            if self.graph_embedding_representation:
                u, v = x[..., 0], x[..., 1]
                scores = jnp.einsum('...ik,...jk->...ij', u, v)
            else:
                scores = x
            
            log_probs, log_probs_neg =  self.log_sig(scores, t), self.log_sig(-scores, t)

            # mask diagonal since it is explicitly not modeled
            # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
            log_probs = index_mul(log_probs, index[..., jnp.arange(log_probs.shape[-1]), jnp.arange(log_probs.shape[-1])], 0.0)
            log_probs_neg = index_mul(log_probs_neg, index[..., jnp.arange(log_probs_neg.shape[-1]), jnp.arange(log_probs_neg.shape[-1])], 0.0)
            return log_probs, log_probs_neg
            
        self.edge_log_probs = jit(edge_log_probs_)

        def sample_g_(p, subk, n_samples):
            '''
            Sample Bernoulli matrix according to matrix of probabilities
                p:   [d, d]
                out: [n_samples, d, d]
            '''
            g_samples = self.vec_to_mat(random.bernoulli(
                subk, p=self.mat_to_vec(p),
                 shape=(n_samples, self.n_vars * self.n_vars))).astype(jnp.int32)

            # mask diagonal since it is explicitly not modeled
            g_samples = index_mul(g_samples, index[..., jnp.arange(p.shape[-1]), jnp.arange(p.shape[-1])], 0)

            return g_samples

        self.sample_g = jit(sample_g_, static_argnums=(2,))
        # [J, d, d], [J, ], [1, ] -> [n_samples, J, d, d]
        self.eltwise_sample_g = jit(vmap(sample_g_, (0, 0, None), 1), static_argnums=(2,))

        def latent_log_prob(single_g, single_x, t):
            '''
            log p(G | U, V)
                single_g:   [d, d]    
                single_x:   [d, k, 2]
                out:        [1,]
            Defined for gradient with respect to `single_x`, i.e. U and V
            '''
            # [d, d], [d, d]
            log_p, log_1_p = self.edge_log_probs(single_x, t)

            # [d, d]
            log_prob_g_ij = single_g * log_p + (1 - single_g) * log_1_p

            # [1,] # diagonal is masked inside `edge_log_probs`
            log_prob_g = jnp.sum(log_prob_g_ij)

            return log_prob_g
        
        # [n_graphs, d, d], [n_particles, d, k, 2] -> [n_graphs]
        self.eltwise_latent_log_prob = jit(vmap(latent_log_prob, (0, None, None), 0))
        # [n_graphs, d, d], [n_particles, d, k, 2] -> [n_graphs, n_particles]
        self.double_eltwise_latent_log_prob = jit(vmap(self.eltwise_latent_log_prob, (None, 0, None), 1))

        # [d, d], [d, k, 2] -> [d, k, 2]
        grad_latent_log_prob = grad(latent_log_prob, 1)

        # [n_graphs, d, d], [d, k, 2] -> [n_graphs, d, k, 2]
        self.eltwise_grad_latent_log_prob = jit(vmap(grad_latent_log_prob, (0, None, None), 0))
        # self.eltwise_grad_latent_log_prob = vmap(grad_latent_log_prob, (0, None, None), 0)

        # [n_graphs, d, d], [d, d], [1,], [1,] -> [n_graphs]
        self.eltwise_log_joint_prob = jit(vmap(self.target_log_joint_prob, (0, None, None, None), 0))

        # only used for metrics computed on the fly
        # [L, d, d], [L, d, d], [1,] -> [L]
        self.double_eltwise_log_joint_prob_no_batch = jit(vmap(self.target_log_joint_prob_no_batch, (0, 0, None), 0))

        '''
        Data likelihood gradient estimators
        Refer to https://arxiv.org/abs/1906.10652
        '''
        def __particle_to_soft_graph__(x, eps, t):
            """ 
            Gumbel-softmax / discrete distribution using Logistic(0,1) samples `eps`

                x:    [n_vars, n_dim, 2]
                eps:  [n_vars, n_vars]
            
                out:  [n_vars, n_vars]
            """
            if self.graph_embedding_representation:
                scores = jnp.einsum('...ik,...jk->...ij', x[..., 0], x[..., 1])
            else:
                scores = x
            # probs = 1 / (1 + jnp.exp(- alpha * scores))

            # soft reparameterization using gumbel-softmax/concrete distribution
            # sig_expr = self.tau(t) * (jnp.log(u) - jnp.log(1 - u) + self.alpha(t) * scores)
            # eps ~ Logistic(0,1)
            soft_graph = sigmoid(self.tau(t) * (eps + self.alpha(t) * scores))

            # mask diagonal since it is explicitly not modeled
            soft_graph = index_mul(soft_graph, index[..., jnp.arange(soft_graph.shape[-1]), jnp.arange(soft_graph.shape[-1])], 0.0)
            return soft_graph

        def __particle_to_hard_graph__(x, eps, t):
            """ 
            Bernoulli sample of G using probabilities implied by x

                x:    [n_vars, n_dim, 2]
                eps:  [n_vars, n_vars]
            
                out:  [n_vars, n_vars]
            """
            if self.graph_embedding_representation:
                scores = jnp.einsum('...ik,...jk->...ij', x[..., 0], x[..., 1])
            else:
                scores = x
            # probs = 1 / (1 + jnp.exp(- alpha * scores))

            # simply take hard limit of sigmoid in gumbel-softmax/concrete distribution
            hard_graph = ((self.tau(t) * (eps + self.alpha(t) * scores)) > 0.0).astype(jnp.float64)

            # mask diagonal since it is explicitly not modeled
            hard_graph = index_mul(hard_graph, index[..., jnp.arange(hard_graph.shape[-1]), jnp.arange(hard_graph.shape[-1])], 0.0)
            return hard_graph

        #
        # w.r.t x (latent embeddings for graph)
        #

        if self.grad_estimator_x == 'score':
            
            # does not use d/dG log p(theta, D | G) (i.e. applicable when not defined)

            def grad_x_likelihood_sf_(single_x, single_theta, single_sf_baseline, t, subk, b):
                '''
                Score function estimator for gradient d/dZ of expectation of 
                p(theta, D | G) w.r.t G|Z  (here denoted `x`)

                Uses same G for both expectations
                    single_x:           [d, k, 2]
                    single_theta:       [d, d]
                    single_sf_baseline: [1, ]
                    out:                [d, k, 2], [1, ]

                '''
                # [d, d]
                p = self.edge_probs(single_x, t)

                # [n_grad_mc_samples, d, d]
                subk, subk_ = random.split(subk)
                g_samples = self.sample_g(p, subk_, self.n_grad_mc_samples)

                # same MC samples for numerator and denominator
                n_mc_numerator = self.n_grad_mc_samples
                n_mc_denominator = self.n_grad_mc_samples
                latent_dim = self.n_vars * self.n_vars

                # [n_mc_numerator, ] 
                subk, subk_ = random.split(subk)
                logprobs_numerator = self.eltwise_log_joint_prob(g_samples, single_theta, b, subk_)
                logprobs_denominator = logprobs_numerator

                # variance_reduction
                logprobs_numerator_adjusted = lax.cond(
                    self.score_function_baseline <= 0.0,
                    lambda _: logprobs_numerator,
                    lambda _: logprobs_numerator - single_sf_baseline,
                    operand=None)

                # [n_vars * n_dim * 2, n_mc_numerator]
                if self.graph_embedding_representation:
                    grad_x = self.eltwise_grad_latent_log_prob(g_samples, single_x, t)\
                        .reshape(self.n_grad_mc_samples, self.n_vars * self.n_dim * 2)\
                        .transpose((1, 0))
                else:
                    grad_x = self.eltwise_grad_latent_log_prob(g_samples, single_x, t)\
                        .reshape(self.n_grad_mc_samples, self.n_vars * self.n_vars)\
                        .transpose((1, 0))

                # stable computation of exp/log/divide
                # [n_vars * n_dim * 2, ]  [n_vars * n_dim * 2, ]
                log_numerator, sign = logsumexp(a=logprobs_numerator_adjusted, b=grad_x, axis=1, return_sign=True)

                # []
                log_denominator = logsumexp(logprobs_denominator, axis=0)

                # [n_vars * n_dim * 2, ]
                stable_sf_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

                # [n_vars, n_dim, 2]
                if self.graph_embedding_representation:
                    stable_sf_grad_shaped = stable_sf_grad.reshape(self.n_vars, self.n_dim, 2)
                else:
                    stable_sf_grad_shaped = stable_sf_grad.reshape(self.n_vars, self.n_vars)

                # update baseline
                single_sf_baseline = (self.score_function_baseline * logprobs_numerator.mean(0) +
                                 (1 - self.score_function_baseline) * single_sf_baseline)

                return stable_sf_grad_shaped, single_sf_baseline

            self.grad_x_likelihood = jit(grad_x_likelihood_sf_)

        elif 'reparam' in self.grad_estimator_x:

            # assumes that d/dG log p(theta, D | G) exists

            def __log_joint_prob_soft__(single_x, single_theta, eps, t, b, subk):
                '''
                This is the composition of 
                    log p(theta, D | G)
                and
                    G(Z, U)

                    single_x:           [d, k, 2]
                    single_theta:       [d, d]
                    eps:                [d, d] ~ Logistic(0,1)
                    t:                  [1,]
                    b:                  [b,]
                    subk:               [1,]

                    out:                [1, ]

                '''
                soft_g_sample = __particle_to_soft_graph__(single_x, eps, t)

                return self.target_log_joint_prob(soft_g_sample, single_theta, b, subk)
            
            def __log_joint_prob_hard__(single_x, single_theta, eps, t, b, subk):
                '''
                This is the composition of 
                    log p(theta, D | G)
                and
                    G(Z, U)

                    single_x:           [d, k, 2]
                    single_theta:       [d, d]
                    eps:                [d, d] ~ Logistic(0,1)
                    t:                  [1,]
                    b:                  [b,]
                    subk:               [1,]

                    out:                [1, ]

                '''
                hard_g_sample = __particle_to_hard_graph__(single_x, eps, t)

                return self.target_log_joint_prob(hard_g_sample, single_theta, b, subk)

            # [d, k, 2], [d, d], [n_graphs, d, d] -> [n_graphs]
            # i.e. parallelized over G (i.e randomness `eps`); Z and theta fixed
            eltwise_log_joint_prob_soft = jit(vmap(__log_joint_prob_soft__, (None, None, 0, None, None, None), 0))
            eltwise_log_joint_prob_hard = jit(vmap(__log_joint_prob_hard__, (None, None, 0, None, None, None), 0))

            # [d, k, 2], [d, d], [d, d] -> [d, k, 2]
            grad_x_log_joint_prob_soft = grad(__log_joint_prob_soft__, 0)
            grad_x_log_joint_prob_hard = grad(__log_joint_prob_hard__, 0)

            # [d, k, 2], [d, d], [n_graphs, d, d] -> [n_graphs, d, k, 2]
            # i.e. parallelized over randomness `eps`
            eltwise_grad_x_log_joint_prob_soft = jit(vmap(grad_x_log_joint_prob_soft, (None, None, 0, None, None, None), 0))
            eltwise_grad_x_log_joint_prob_hard = jit(vmap(grad_x_log_joint_prob_hard, (None, None, 0, None, None, None), 0))

            
            def grad_x_likelihood_reparam_(single_x, single_theta, single_sf_baseline, t, subk, b):
                '''
                Reparameterization estimator for gradient d/dZ of expectation of 
                p(theta, D | G) w.r.t G|Z  (here denoted `x`)

                Uses same G for both expectations
                    single_x:           [d, k, 2]
                    single_theta:       [d, d]
                    single_sf_baseline: [1, ]
                    out:                [d, k, 2], [1, ]

                '''               
                # same MC samples for numerator and denominator
                n_mc_numerator = self.n_grad_mc_samples
                n_mc_denominator = self.n_grad_mc_samples

                # sample Logistic(0,1) as randomness in reparameterization
                subk, subk_ = random.split(subk)
                eps = random.logistic(subk_, shape=(self.n_grad_mc_samples, self.n_vars, self.n_vars))                

                # [n_mc_numerator, ]
                # since we don't backprop per se, it leaves us with the option of having
                # `soft` and `hard` versions for evaluating the non-grad p(.))
                subk, subk_ = random.split(subk)
                if self.grad_estimator_x == 'reparam_soft':
                    logprobs_numerator = eltwise_log_joint_prob_soft(single_x, single_theta, eps, t, b, subk_)
                elif self.grad_estimator_x == 'reparam_hard':
                    logprobs_numerator = eltwise_log_joint_prob_hard(single_x, single_theta, eps, t, b, subk_)
                else:
                    raise KeyError('Invalid reparameterization `grad_estimator_x`. Choose `reparam_soft` or `reparam_hard`.')

                logprobs_denominator = logprobs_numerator

                # [n_mc_numerator, n_vars, n_dim, 2]
                # d/dx log p(theta, D | G(x, eps)) for a batch of `eps` samples
                # use the same minibatch of data as for other log prob evaluation (if using minibatching)
                grad_x = eltwise_grad_x_log_joint_prob_soft(single_x, single_theta, eps, t, b, subk_)

                # stable computation of exp/log/divide
                # [n_vars, n_dim, 2], [n_vars, n_dim, 2]
                if self.graph_embedding_representation:
                    log_numerator, sign = logsumexp(a=logprobs_numerator[:, None, None, None], b=grad_x, axis=0, return_sign=True)
                else:
                    log_numerator, sign = logsumexp(a=logprobs_numerator[:, None, None], b=grad_x, axis=0, return_sign=True)

                # []
                log_denominator = logsumexp(logprobs_denominator, axis=0)

                # [n_vars, n_dim, 2]
                stable_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

                return stable_grad, single_sf_baseline

            self.grad_x_likelihood = jit(grad_x_likelihood_reparam_)

        else:
            raise ValueError('Unknown gradient estimator `grad_estimator_x`')
    
        self.eltwise_grad_x_likelihood = jit(vmap(self.grad_x_likelihood, (0, 0, 0, None, 0, None), (0, 0)))

        #
        # w.r.t theta (graph parameters)
        #

        # assumes that d/dtheta log p(theta, D | z) exists
        # [d, d], [d, d], [1,], [1,] -> [d, d]
        grad_theta_log_joint_prob = grad(self.target_log_joint_prob, 1)

        # [n_graphs, d, d], [d, d], [1,], [1,] -> [n_graphs, d, d]
        # i.e. parallelized over Z, not theta
        eltwise_grad_theta_log_joint_prob = jit(vmap(grad_theta_log_joint_prob, (0, None, None, None), 0))

        if self.grad_estimator_theta == 'hard':

            # uses hard samples of G; reparameterization like for d/dz is also possible

            def grad_theta_likelihood_hard_(single_x, single_theta, t, subk, b):
                '''
                Score function estimator for gradient of expectation over 
                p(theta, D | G) w.r.t latent variables Z (here denoted `x`)

                Uses same G for both expectations
                    single_x:           [d, k, 2]
                    single_theta:       PyTree

                    out:                PyTree

                '''

                # [d, d]
                p = self.edge_probs(single_x, t)

                # [n_grad_mc_samples, d, d]
                g_samples = self.sample_g(p, subk, self.n_grad_mc_samples)

                # same MC samples for numerator and denominator
                n_mc_numerator = self.n_grad_mc_samples
                n_mc_denominator = self.n_grad_mc_samples

                # [n_mc_numerator, ] 
                subk, subk_ = random.split(subk)
                logprobs_numerator = self.eltwise_log_joint_prob(g_samples, single_theta, b, subk_)
                logprobs_denominator = logprobs_numerator

                # PyTree  shape of `single_theta` with additional leading dimension [n_mc_numerator, ...]
                # d/dtheta log p(theta, D | G) for a batch of G samples
                # use the same minibatch of data as for other log prob evaluation (if using minibatching)
                grad_theta = eltwise_grad_theta_log_joint_prob(g_samples, single_theta, b, subk_)

                # stable computation of exp/log/divide and PyTree compatible
                # sums over MC graph samples dimension to get MC gradient estimate of theta
                # original PyTree shape of `single_theta`
                log_numerator = tree_map(
                    lambda leaf_theta: 
                        logsumexp(a=expand_by(logprobs_numerator, leaf_theta.ndim - 1), b=leaf_theta, axis=0, return_sign=True)[0], 
                    grad_theta)

                # original PyTree shape of `single_theta`
                sign = tree_map(
                    lambda leaf_theta:
                        logsumexp(a=expand_by(logprobs_numerator, leaf_theta.ndim - 1), b=leaf_theta, axis=0, return_sign=True)[1], 
                    grad_theta)

                # []
                log_denominator = logsumexp(logprobs_denominator, axis=0)

                # original PyTree shape of `single_theta`
                stable_grad = tree_multimap(
                    lambda sign_leaf_theta, log_leaf_theta: 
                        (sign_leaf_theta * jnp.exp(log_leaf_theta - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))), 
                    sign, log_numerator)

                return stable_grad


            self.grad_theta_likelihood = jit(grad_theta_likelihood_hard_)

        else:
            raise ValueError('Unknown gradient estimator `grad_estimator_theta`')

        self.eltwise_grad_theta_likelihood = jit(vmap(self.grad_theta_likelihood, (0, 0, None, None, None), 0))

        '''
        Acyclicity constraint
        '''
        # jacobian: particle -> G
        # [d, d, d, k, 2]
        jac_particle_to_soft_graph__ = jacrev(__particle_to_soft_graph__, 0)

        # gradient: G -> h
        # [d, d]
        grad_acyclic_constr__ = grad(acyclic_constr_nograd, 0)

        ###
        # composite functions
        ###

        if self.constraint_prior_graph_sampling is None:
            #
            # Matrix of probabilities only (using gamma(t) for scaling)
            #
            def __constraint__(x, eps, t):
                ''' 
                Constraint evaluation for a single particle
                Used to define gradient of prior
                    x:      [n_vars, n_dim, 2]

                    eps:    [n_vars, n_vars] 
                        (ignored in this option of `constraint_prior_graph_sampling`)

                    out:    [1, ]
                '''
                # [n_vars, n_vars] # diagonal is masked inside `edge_probs`
                p = self.edge_probs(x, t)

                # scale to avoid vanishing gradient
                p = self.gamma(t) * p

                h = acyclic_constr_nograd(p, self.n_vars)
                return h

            grad_constraint = grad(__constraint__, 0)

        elif self.constraint_prior_graph_sampling == 'soft':
            #
            # Samples graphs using Gumbel-softmax in forward and backward pass
            #

            def __constraint__(single_x, single_eps, t):
                """ 
                Evaluates continuous acyclicity constraint using 
                Gumbel-softmax instead of Bernoulli samples

                    single_x:      [n_vars, n_dim, 2]
                    single_eps:    [n_vars, n_vars]
                
                    out:  [1,]
                """
                G = __particle_to_soft_graph__(single_x, single_eps, t)
                h = acyclic_constr_nograd(G, self.n_vars)
                return h

            def grad_constraint_single_eps_(single_x, single_eps, t):
                '''
                Pathwise derivative of d/dZ h(G(Z, eps))
                    single_x:      [n_vars, n_dim, 2]
                    single_eps:    [n_vars, n_vars]
                
                    out:  [1,]
                '''
            
                # dG(Z)/dZ: [d, d, d, k, 2]
                # since mapping [d, k, 2] -> [d, d]
                jac_G_Z = jac_particle_to_soft_graph__(single_x, single_eps, t)

                # G(Z)
                # [d, d]
                single_G = __particle_to_soft_graph__(single_x, single_eps, t)

                # dh(G)/dG
                grad_h_G = grad_acyclic_constr__(single_G, self.n_vars)

                # pathwise derivative
                # sum over all i,j in G (from h back to Z)
                # [d, k, 2]
                if self.graph_embedding_representation:
                    dZ = jnp.sum(grad_h_G[:, :, None, None, None] * jac_G_Z, axis=(0, 1))
                else:
                    dZ = jnp.sum(grad_h_G[:, :, None, None] * jac_G_Z, axis=(0, 1))
                return dZ

            # [n_vars, n_dim, 2], [mc_samples, n_vars, n_vars] -> [mc_samples, n_vars, n_dim, 2]
            # grad_constraint_batch_eps = jit(vmap(grad_constraint_single_eps_, (None, 0, None), 0))
            # autodiff avoids representing juge jacobian (jacrev) in memory (only possible for `soft`)
            grad_constraint_batch_eps = jit(vmap(grad(__constraint__, 0), (None, 0, None), 0))

            def grad_constraint(x, key, t):
                '''
                Monte Carlo estimator of expectation of constraint gradient
                    x:    [n_vars, n_dim, 2]                
                    key:  [1,]             
                    out:  [n_vars, n_dim, 2] 
                '''
                key, subk = random.split(key)
                eps = random.logistic(key, shape=(self.n_acyclicity_mc_samples, self.n_vars, self.n_vars))

                mc_gradient = grad_constraint_batch_eps(x, eps, t).mean(0)

                return mc_gradient

        elif self.constraint_prior_graph_sampling == 'hard':
            #
            # Samples graphs using Gumbel-softmax in backward but /Bernoulli/ in forward pass
            #
            def __constraint__(single_x, single_eps, t):
                """ 
                Evaluates continuous acyclicity constraint using 
                Gumbel-softmax instead of Bernoulli samples

                    single_x:      [n_vars, n_dim, 2]
                    single_eps:    [n_vars, n_vars]
                
                    out:  [1,]
                """
                G = __particle_to_hard_graph__(single_x, single_eps, t)
                h = acyclic_constr_nograd(G, self.n_vars)
                return h

            def grad_constraint_single_eps_(single_x, single_eps, t):
                '''
                Pathwise derivative of d/dZ h(G(Z, eps))
                    single_x:      [n_vars, n_dim, 2]
                    single_eps:    [n_vars, n_vars]
                
                    out:  [1,]
                '''
            
                # dG(Z)/dZ: [d, d, d, k, 2]
                # since mapping [d, k, 2] -> [d, d]
                jac_G_Z = jac_particle_to_soft_graph__(single_x, single_eps, t)

                # G(Z)
                # [d, d]
                single_G = __particle_to_hard_graph__(single_x, single_eps, t)

                # dh(G)/dG
                grad_h_G = grad_acyclic_constr__(single_G, self.n_vars)

                # pathwise derivative
                # sum over all i,j in G (from h back to Z)
                # [d, k, 2]
                if self.graph_embedding_representation:
                    dZ = jnp.sum(grad_h_G[:, :, None, None, None] * jac_G_Z, axis=(0, 1))
                else:
                    dZ = jnp.sum(grad_h_G[:, :, None, None] * jac_G_Z, axis=(0, 1))
                return dZ

            # [n_vars, n_dim, 2], [mc_samples, n_vars, n_vars] -> [mc_samples, n_vars, n_dim, 2]
            grad_constraint_batch_eps = jit(vmap(grad_constraint_single_eps_, (None, 0, None), 0))

            def grad_constraint(x, key, t):
                '''
                Monte Carlo estimator of expectation of constraint gradient
                    x:    [n_vars, n_dim, 2]                
                    key:  [1,]             
                    out:  [n_vars, n_dim, 2] 
                '''
                key, subk = random.split(key)
                eps = random.logistic(key, shape=(self.n_acyclicity_mc_samples, self.n_vars, self.n_vars))

                mc_gradient = grad_constraint_batch_eps(x, eps, t).mean(0)

                return mc_gradient

        else:
            raise ValueError('Invalid value in `constraint_prior_graph_sampling`')

        #
        # Collect functions; same signature for all options
        #
        
        # [n_vars, n_dim, 2], [n_vars, n_vars] -> 1
        self.constraint = jit(__constraint__)

        # [A, n_vars, n_dim, 2], [B, n_vars, n_vars] -> [A, B]
        self.eltwise_constraint = jit(vmap(vmap(__constraint__, (None, 0, None), 0), (0, None, None), 0))
        
        # [n_particles, n_vars, n_dim, 2], [n_particles,]  -> [n_particles, n_vars, n_dim, 2]
        self.eltwise_grad_constraint = jit(vmap(grad_constraint, (0, 0, None), 0))


        '''
        Latent prior 
        p(Z) = exp(- beta(t) * h(G|Z))

        if `latent_prior_std` > 0, additional factor with
        elementwise zero-mean diagonal gaussian 
        '''

        def target_log_prior_particle(single_x, t):
            '''
            log p(U, V) approx. log p(G)
                single_x:   [d, k, 2]
                out:        [1,]
            '''
            # [d, d] # masking is done inside `edge_probs`
            single_soft_g = self.edge_probs(single_x, t)

            # [1, ]
            return self.target_log_prior(single_soft_g)

        # [d, k, 2], [1,] -> [d, k, 2]
        grad_target_log_prior_particle = jit(grad(target_log_prior_particle, 0))

        # [n_particles, d, k, 2], [1,] -> [n_particles, d, k, 2]
        self.eltwise_grad_target_log_prior_particle = jit(
            vmap(grad_target_log_prior_particle, (0, None), 0))


        if self.latent_prior_std > 0.0:        
            # gaussian and acyclicity
            def eltwise_grad_latent_prior_(x, subkeys, t):
                grad_prior_x = self.eltwise_grad_target_log_prior_particle(x, t)
                grad_constraint_x = self.eltwise_grad_constraint(x, subkeys, t)
                return grad_prior_x \
                       - self.beta(t) * grad_constraint_x \
                       - 2.0 * x / (self.latent_prior_std ** 2.0)
        else:
            # only acyclicity
            def eltwise_grad_latent_prior_(x, subkeys, t):
                grad_prior_x = self.eltwise_grad_target_log_prior_particle(x, t)
                grad_constraint_x = self.eltwise_grad_constraint(x, subkeys, t)
                return grad_prior_x \
                       - self.beta(t) * grad_constraint_x

        # prior p(W)
        self.eltwise_grad_latent_prior = jit(eltwise_grad_latent_prior_)
        # self.eltwise_grad_latent_prior = eltwise_grad_latent_prior_

        '''
        Kernel eval and grad
        '''  

        def f_kernel_(x_latent, x_theta, y_latent, y_theta, h_latent, h_theta, t):
            return self.kernel.eval(
                x_latent=x_latent, x_theta=x_theta,
                y_latent=y_latent, y_theta=y_theta,
                h_latent=h_latent, h_theta=h_theta,
                alpha=self.alpha(t))

        self.f_kernel = jit(f_kernel_)
        grad_kernel_x =     jit(grad(self.f_kernel, 0))
        grad_kernel_theta = jit(grad(self.f_kernel, 1))
        self.eltwise_grad_kernel_x =     jit(vmap(grad_kernel_x,     (0, 0, None, None, None, None, None), 0))
        self.eltwise_grad_kernel_theta = jit(vmap(grad_kernel_theta, (0, 0, None, None, None, None, None), 0))

        '''
        Define single SVGD particle update for jit and vmap
        '''

        #
        # x (latent embeddings)
        #
        def x_update(single_x, single_theta, kxx_for_x, x, theta, grad_log_prob_x, h_latent, h_theta, t):
        
            # compute terms in sum
            if self.graph_embedding_representation:
                weighted_gradient_ascent = kxx_for_x[..., None, None, None] * grad_log_prob_x
            else:
                weighted_gradient_ascent = kxx_for_x[..., None, None] * grad_log_prob_x
            
            repulsion = self.eltwise_grad_kernel_x(x, theta, single_x, single_theta, h_latent, h_theta, t)

            # average and negate
            return - (weighted_gradient_ascent + repulsion).mean(axis=0)

        #
        # theta 
        #
        def theta_update(single_x, single_theta, kxx_for_x, x, theta, grad_log_prob_theta, h_latent, h_theta, t):
        
            # compute terms in sum
            weighted_gradient_ascent = tree_map(
                lambda leaf_theta_grad: 
                    expand_by(kxx_for_x, leaf_theta_grad.ndim - 1) * leaf_theta_grad, 
                grad_log_prob_theta)
            
            repulsion = self.eltwise_grad_kernel_theta(x, theta, single_x, single_theta, h_latent, h_theta, t)

            # average and negate
            return  tree_multimap(
                lambda grad_asc_leaf, repuls_leaf: 
                    - (grad_asc_leaf + repuls_leaf).mean(axis=0), 
                weighted_gradient_ascent, 
                repulsion)

        self.parallel_update_x =     jit(vmap(x_update,     (0, 0, 1, None, None, None, None, None, None), 0))
        self.parallel_update_theta = jit(vmap(theta_update, (0, 0, 1, None, None, None, None, None, None), 0))

        #
        # one loop iteration, to be batch processed
        #
        def loop_iter_(x, theta, t, key, b, sf_baseline):

            # make sure same bandwith is used for all calls to k(x,x') if the median heuristic is applied
            h_latent = self.kernel.h_latent
            h_theta = self.kernel.h_theta
            n_particles = x.shape[0]

            # d/dtheta log p(theta, D | z)
            key, subk = random.split(key)
            dtheta_log_prob = self.eltwise_grad_theta_likelihood(x, theta, t, subk, b)

            # d/dz log p(theta, D | z)
            key, *batch_subk = random.split(key, n_particles + 1)
            dx_log_likelihood, sf_baseline = self.eltwise_grad_x_likelihood(x, theta, sf_baseline, t, jnp.array(batch_subk), b)

            # d/dz log p(z) (acyclicity)
            key, *batch_subk = random.split(key, n_particles + 1)
            dx_log_prior = self.eltwise_grad_latent_prior(x, jnp.array(batch_subk), t)

            # d/dz log p(z, theta, D) = d/dz log p(z)  + log p(theta, D | z) 
            dx_log_prob = dx_log_prior + dx_log_likelihood
           
            # k((x, theta), (x, theta))
            kxx = self.f_kernel(x, theta, x, theta, h_latent, h_theta, t)

            # transformation phi(x)
            phi_x = self.parallel_update_x(x, theta, kxx, x, theta, dx_log_prob, h_latent, h_theta, t)
            phi_theta = self.parallel_update_theta(x, theta, kxx, x, theta, dtheta_log_prob, h_latent, h_theta, t)

            if self.graph_embedding_representation:
                if self.fix_rotation != "not":
                    # do not update u_0 and v_0 as they are fixed
                    phi_x = index_update(phi_x, index[:, 0, :, :], 0.0)

            return phi_x, phi_theta, key, sf_baseline

        self.batch_loop_iter = jit(vmap(loop_iter_, (0, 0, None, 0, 0, 0), (0, 0, 0, 0)))


        self.has_init_core_functions = True

    def sample_particles(self, *, n_steps, init_particles_x, init_particles_theta, key,
                         eval_metrics=[], tune_metrics=[], metric_every=1, iter0=0):
        """
        Deterministically transforms particles as provided by `init_particles`
        to minimize KL to target using SVGD
        """
        h_is_none = self.kernel.h_latent == -1.0 and self.kernel.h_theta == -1.0

        batch_x = init_particles_x
        batch_theta = init_particles_theta
      
        # initialize particles
        batch_size = batch_x.shape[0]

        if self.graph_embedding_representation:
            if self.fix_rotation == 'parallel':
                batch_x = index_update(batch_x, index[:, :, 0, :, :], 1.0 * self.latent_prior_std)

            elif self.fix_rotation == 'orthogonal':
                pm_ones = jnp.where(jnp.arange(batch_x.shape[-2]) % 2, -1.0, 1.0).reshape(1, 1, -1)
                batch_x = index_update(batch_x, index[:, :, 0, :, 0],   pm_ones * self.latent_prior_std)
                batch_x = index_update(batch_x, index[:, :, 0, :, 1], - pm_ones * self.latent_prior_std)

            elif self.fix_rotation  == 'not':
                pass 

            else:
                raise ValueError('Invalid `fix_rotation` keyword')

        # initialize score function baseline (one for each particle)
        batch_sf_baseline = jnp.zeros(batch_x.shape[0:2])
        
        # jit core functions
        if not self.has_init_core_functions:
            self.init_core_functions()

        # init optimizer
        if self.optimizer['name'] == 'gd':
            opt_init, opt_update, get_params = optimizers.sgd(self.optimizer['stepsize']/ 10.0) # comparable scale for tuning
        elif self.optimizer['name'] == 'momentum':
            opt_init, opt_update, get_params = optimizers.momentum(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adagrad':
            opt_init, opt_update, get_params = optimizers.adagrad(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adam':
            opt_init, opt_update, get_params = optimizers.adam(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'rmsprop':
            opt_init, opt_update, get_params = optimizers.rmsprop(self.optimizer['stepsize'])
        else:
            raise ValueError()

        batch_opt_init = jit(vmap(opt_init, 0, 0))
        batch_opt_update = jit(vmap(opt_update, (None, 0, 0), 0))
        batch_get_params = jit(vmap(get_params, 0, 0))

        batch_opt_state_x = batch_opt_init(batch_x)
        batch_opt_state_theta = batch_opt_init(batch_theta)
        batch_keys = random.split(key, batch_size)

        '''execute particle updates'''
        it = tqdm.tqdm(range(iter0, n_steps + iter0), desc='SVGD', disable=not self.verbose)
        for t in it:

            # loop iteration in batch
            batch_ids = jnp.arange(batch_size)
            batch_x = batch_get_params(batch_opt_state_x)
            batch_theta = batch_get_params(batch_opt_state_theta)

            batch_phi_x, batch_phi_theta, batch_keys, batch_sf_baseline = self.batch_loop_iter(
                batch_x, batch_theta, t, batch_keys, batch_ids, batch_sf_baseline)

            # apply transformation
            # `x += stepsize * phi`; the phi returned is negated for SVGD
            batch_opt_state_x = batch_opt_update(t, batch_phi_x, batch_opt_state_x)
            batch_opt_state_theta = batch_opt_update(t, batch_phi_theta, batch_opt_state_theta)

            # check if something went wrong
            phi_x_isnan = jnp.any(jnp.isnan(batch_phi_x))
            phi_theta_isnan = tree_reduce(lambda leaf, c: leaf or c, 
                tree_map(lambda leaf: jnp.isnan(leaf).any(), batch_phi_theta), initializer=False)
            
            if phi_x_isnan:
                raise SVGDNaNErrorLatent()
                exit()

            if phi_theta_isnan:
                raise SVGDNaNErrorTheta()
                exit()

            # evaluate
            if not t % metric_every:
                batch_x = batch_get_params(batch_opt_state_x)
                batch_theta = batch_get_params(batch_opt_state_theta)
                step = t // metric_every
                params = dict(
                    batch_keys=batch_keys,
                    step=step,
                    t=t,
                    batch_x=batch_x,
                    batch_theta=batch_theta,
                    batch_phi_x=batch_phi_x,
                    batch_phi_theta=batch_phi_theta,
                    alpha=self.alpha(t),
                    beta=self.beta(t),
                    tau=self.tau(t),
                )

                if eval_metrics:
                    metrics = ' | '.join(
                        f(params) for f in eval_metrics
                    )
                    it.set_description('SVGD | ' + metrics)

                if tune_metrics:
                    for f in tune_metrics:
                        f(params)

        # evaluate metrics once more at the end
        if tune_metrics or eval_metrics:
            step = t // metric_every
            batch_x = batch_get_params(batch_opt_state_x)
            batch_theta = batch_get_params(batch_opt_state_theta)
            params = dict(
                batch_keys=batch_keys,
                step=step,
                t=n_steps + iter0,
                batch_x=batch_x,
                batch_theta=batch_theta,
                batch_phi_x=batch_phi_x,
                batch_phi_theta=batch_phi_theta,
                alpha=self.alpha(t),
                beta=self.beta(t),
                tau=self.tau(t),
            )
            if eval_metrics:
                metrics = ' | '.join(
                    f(params) for f in eval_metrics
                )
                it.set_description('SVGD | ' + metrics)
                # not the same as flush=True, so metrics might not match printed results

            if tune_metrics:
                for f in tune_metrics:
                    f(params)
        
        batch_x_final = batch_get_params(batch_opt_state_x)
        batch_theta_final = batch_get_params(batch_opt_state_theta)
        return batch_x_final, batch_theta_final


    """
    Helper functions for progress tracking
    """

    def make_metrics(self, variants_target, eltwise_log_likelihood):

        n_variants = len(variants_target)

        def neg_log_likelihood_metric(params):
            '''
            Average log posterior predictve on held-out data with joint (G, theta) samples
            '''
            batch_x = params['batch_x']
            batch_theta = tree_unzip_leading(params['batch_theta'], len(batch_x))
            batch_hard_g = self.particle_to_hard_g(batch_x)
        
            mean_ll = jnp.array([
                neg_ave_log_likelihood(
                    dist=particle_joint_empirical(hard_g, theta),
                    eltwise_log_joint_target=eltwise_log_likelihood,
                    x=jnp.array(variants_target[b % n_variants].x_ho)
                )
                for b, (hard_g, theta) in enumerate(zip(batch_hard_g, batch_theta))
            ]).mean()

            mean_ll_mixt = jnp.array([
                neg_ave_log_likelihood(
                    dist=particle_joint_mixture(hard_g, theta, 
                        lambda ws_, thetas_: self.double_eltwise_log_joint_prob_no_batch(ws_, thetas_, b)),
                    eltwise_log_joint_target=eltwise_log_likelihood,
                    x=jnp.array(variants_target[b % n_variants].x_ho)
                )
                for b, (hard_g, theta) in enumerate(zip(batch_hard_g, batch_theta))
            ]).mean()

            return 'neg LL [empirical]: {:8.04f}  [mixt]:  {:8.04f}'.format(mean_ll, mean_ll_mixt)


        def edge_belief_metric(params):
            '''
            L1 edge belief
            '''
            batch_x = params['batch_x']
            batch_theta = tree_unzip_leading(params['batch_theta'], len(batch_x))
            batch_hard_g = self.particle_to_hard_g(batch_x)
        
            mean_eb = jnp.array([
                l1_edge_belief(
                    dist=(particle_joint_empirical(hard_g, theta)[0], particle_joint_empirical(hard_g, theta)[2]),
                    g=jnp.array(variants_target[b % n_variants].g)
                )
                for b, (hard_g, theta) in enumerate(zip(batch_hard_g, batch_theta))
            ]).mean()

            mean_eb_mixt = jnp.array([
                l1_edge_belief(
                    dist=(particle_joint_mixture(hard_g, theta, 
                            lambda ws_, thetas_: self.double_eltwise_log_joint_prob_no_batch(ws_, thetas_, b))[0],
                          particle_joint_mixture(hard_g, theta, 
                            lambda ws_, thetas_: self.double_eltwise_log_joint_prob_no_batch(ws_, thetas_, b))[2]),
                    g=jnp.array(variants_target[b % n_variants].g)
                )
                for b, (hard_g, theta) in enumerate(zip(batch_hard_g, batch_theta))
            ]).mean()

            return 'L1 [empirical]: {:8.04f}  [mixt]:  {:8.04f}'.format(mean_eb, mean_eb_mixt)

        
        def cyclic_graph_count_hard(params):
            '''
            Computes number of graphs that are cyclic amongst hard graphs implied by x
            '''
            batch_x = params['batch_x']
            batch_hard_g = self.particle_to_hard_g(batch_x)

            mean_unique_cyclic = 0.0
            for hard_g in batch_hard_g:
                ids = bit2id(hard_g)
                unique_hard_g = id2bit(onp.unique(ids, axis=0), self.n_vars)
                unique_dag_count = (elwise_acyclic_constr_nograd(unique_hard_g, self.n_vars) == 0).sum()
                mean_unique_cyclic += unique_hard_g.shape[0] - unique_dag_count
            
            mean_unique_cyclic /= len(batch_hard_g)

            return 'Unique cyclic: {:4.0f}'.format(mean_unique_cyclic)

        def unique_graph_count_hard(params):
            '''
            Computes number of unique graphs implied by x
            '''
            batch_x = params['batch_x']
            batch_hard_g = self.particle_to_hard_g(batch_x)

            mean_n_unique = jnp.array([
                len(onp.unique(bit2id(hard_g), axis=0))
                for hard_g in batch_hard_g
            ]).mean()

            return 'Unique: {:4.0f}'.format(mean_n_unique)

        return dict(
            neg_log_likelihood_metric=neg_log_likelihood_metric,
            edge_belief_metric=edge_belief_metric,
            cyclic_graph_count_hard=cyclic_graph_count_hard,
            unique_graph_count_hard=unique_graph_count_hard,
        )