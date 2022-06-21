import jax.numpy as jnp
import jax.random as random
from dibs.target import make_graph_model

from jax import jit
from jax.scipy.special import gammaln

import gpjax as gpx
import optax as ox

n_inv_gamma = lambda mu, l, alpha, beta: lambda x, sigma_sq: jnp.sqrt(l/(sigma_sq*2*jnp.pi))*(beta**alpha)*(1/jnp.exp(gammaln(alpha)))*((1/sigma_sq)**(alpha+1))*jnp.exp(-(2*beta+l*((x-mu)**2))/(2*sigma_sq))

#TODO Initialize GP hyperparameters


class GaussianProcess:
    def __init__(self, *, n_vars, graph_prior_str="sf", kernel=gpx.RBF(), obs_noise=0.1, n_samples=None):
        super(GaussianProcess, self).__init__()

        if n_samples is None:
            #TODO think of a proper way to handle the changing number of data
            raise
        
        self.n_vars = n_vars
        self.graph_dist = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str) # initialized from make_graph_model in target.py
        self.obs_noise = obs_noise
        self.n_samples = n_samples
        
        # For root nodes R(G)
        #TODO Find out how to match alpha, beta to get 1 sample in JAX
        #TODO Where is the Gamma distribution sampler in JAX taking 2 params?
        self.nr_sigma_sq_sample = lambda key, alpha, beta: 1/jax.random.gamma(key, jnp.array((alpha, beta)))
        self.nr_x_sample = lambda key, mu, l, sigma_sq: mu + jnp.sqrt(sigma_sq/l)*jax.random.normal(key)
        
        # For non-root nodes NR(G)
        #TODO Make a mechanism that takes into account structure of G
        self.kernel=kernel
        self.gp_prior = gpx.Prior(kernel=kernel)
        self.gp_likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.n_samples)
        self.gp_posterior = self.gp_likelihood * self.gp_prior

        # Assuming MAP estimate for GP hyperparams
        # Might be worth it to sample them 
        self.gp_hparams, self.trainable, self.constrainer, self.unconstrainer = gpx.initialise(self.gp_posterior)

        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)

        #TODO Should we sample GPs or use the closed form predictive mean
        #TODO Think about the fact that the GP kernel depends on graph structure
        self.gp_sample = lambda key, theta, x: None #TODO

        self.gp_pred_mean = lambda key, theta, x: None #TODO

        self.eltwise_gp_pred_mean = None #TODO

        self.opt = ox.adam(learning_rate=0.01)

    def update_r_hyperparams(self, x, g):
        # Maximize over hyperparameter space?
        pass


    def update_gp_hyperparams(self, x, g):
        #TODO Technically each of the hyperparameters have some Gamma prior,
        # Check Appendix D.5 for details

        #TODO The kernel depends on the graph structure

        # Convert positive hyperparams to unconstrained optimization space
        opt_params = gpx.transform(self.gp_hparams, self.unconstrainer)
        
        # Right now just using the same package as in GPJax tutorial
        #TODO Use jax optimizers like in inference/svgd.py later
        updated_params = gpx.fit(log_likelihood(x, g, interv_targets, negative=True),
                                 opt_params,
                                 self.trainable,
                                 self.opt,
                                 n_iters=500)
        self.gp_hparams = gpx.transform(updated_params, self.constrainer)


    def get_theta_shape(self, *, n_vars):
        """Returns tree shape of the parameters of the GP

        Args:
            n_vars (int): number of variables in model
        
        """
        pass

    def sample_parameters():
        """Samples batch of random parameters given dimensions of graph from :math: `p(\\Theta | G)`

        """
        pass

    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):

        if interv is None:
            interv = {}
        if toporder is None:
            toporder = g.topological_sorting()

        n_vars = len(g.vs)
        x = jnp.zeros((n_samples, n_vars))

        key, subk = random.split(key)
        z_r = None #TODO
        z_nr = None #TODO

        g_mat = graph_to_mat(g)

        for j in toporder:

            if j in interv.keys():
                x = x.at[index[:, j]].set(interv[j])
                continue

            parents = g_mat[:, j].reshape(1, -1)

            has_parents = parents.sum() > 0

            if has_parents:
                x_msk = x * parents

                means = self.eltwise_gp_pred_mean(theta, x_msk)

                x = x.at[index[:, j]].set(means[:, j]) + z_nr[:, j]
            else:
                x = x.at[index[:, j]].set(z_r[:, j])

        return x

    def log_prob_parameters():
        # p(Theta|G)
        # Ideally the theta should be GP hyperparams + root node hyperparams
        pass

    def log_gp_likelihood(self, *, x, g, interv_targets, negative=True):
        # Log marginal likelihood of non-root nodes
        #TODO Handle intervention target case
        
        all_x_msk = x[None] * g.T[:, None]

        return self.gp_posterior.marginal_log_likelihood(all_x_msk, self.constrainer, negative)

    def log_root_likelihood(self, *, x, g, interv_targets):
        #TODO log marginal likelihood of root nodes

        pass
    

    def log_likelihood(self, *, x, g, interv_targets, negative=False):
        # negative = True for optimization in update_gp_hparams
        from jax.scipy.stats import norm as jax_normal

        nr_mll = self.log_gp_likelihood(x=x, g=g, interv_targets=interv_targets, negative=negative)

        #TODO Create mask that only selects root nodes for r_mll below
        r_sigmasqs = 1.0
        r_means = 0.0
        r_mll = jnp.sum(jnp.where(interv_targets[None, ...], 0.0,
                                  jax_normal.logpdf(x=x, loc=r_means, scale=jnp.sqrt(r_sigmasqs)))
                         )
        
        return nr_mll + r_ll
    
    def log_graph_prior(self, g_prob):
        return self.graph_dist.unnormalized_log_prob_soft(soft_g=g_prob)

    def observational_log_joint_prob(self, x, g, theta):
        log_prob_theta = self.log_prob_parameters(g=g, theta=theta)
        log_likelihood = self.log_likelihood(g=g, theta=theta, x=x, interv_targets=self.no_interv_tagets)
        return log_prob_theta + log_likelihood

    def observational_log_marginal_prob(self, g, _, x, rng, negative=False):
        # P(D|G) : Required for MarginalDiBS computation
        return self.log_likelihood(x=x, g=g, interv_targets=self.no_interv_targets, negative=negative)
        
