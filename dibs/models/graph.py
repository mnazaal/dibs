import igraph as ig
import random as pyrandom
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
from abc import abstractmethod

from dibs.graph_utils import mat_to_graph, graph_to_mat, mat_is_dag

@dataclass
class GraphParameters:
    n_vars:     int
    n_edges_per_node: int

    
class GraphDistribution(GraphParameters):
    @abstractmethod
    def sample_G(self, key, return_mat):
        """Samples DAG

        Args:
            key (ndarray): rng
            return_mat (bool): if ``True``, returns adjacency matrix of shape ``[n_vars, n_vars]``

        Returns:
            ``iGraph.graph`` / ``jnp.array``:
            DAG
        """

    @abstractmethod
    def unnormalized_log_prob_single(self, *, g, j):
        """
        Computes :math:`\\log p(G_j)` up the normalization constant

        Args:
            g (iGraph.graph): graph
            j (int): node index:

        Returns:
            unnormalized log probability of node family of :math:`j`

        """

    @abstractmethod
    def unnormalized_log_prob(self, *, g):
        """
        Computes :math:`\\log p(G)` up the normalization constant

        Args:
            g (iGraph.graph): graph

        Returns:
            unnormalized log probability of :math:`G`

        """

    @abstractmethod
    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        Computes :math:`\\log p(G)` up the normalization constant
        where :math:`G` is the matrix of edge probabilities

        Args:
            soft_g (ndarray): graph adjacency matrix, where entries
                may be probabilities and not necessarily 0 or 1

        Returns:
            unnormalized log probability corresponding to edge probabilities in :math:`G`

        """


class ErdosReniDAGDistribution(GraphDistribution):
    """
    Randomly oriented Erdos-Reni random graph model with i.i.d. edge probability.
    The pmf is defined as

    :math:`p(G) \\propto p^e (1-p)^{\\binom{d}{2} - e}`

    where :math:`e` denotes the total number of edges in :math:`G`
    and :math:`p` is chosen to satisfy the requirement of sampling ``n_edges_per_node``
    edges per node in expectation.

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable in expectation

    """
    def __init__(self, n_vars, n_edges_per_node=2):
        super().__init__(n_vars, n_edges_per_node)
        self.n_edges = n_vars*n_edges_per_node
        self.p_edge = self.n_edges / ((n_vars * (n_vars - 1)) / 2)

    def sample_G(self, key, return_mat=False):
        key, subk = random.split(key)
        mat = random.bernoulli(subk, p=self.p_edge, shape=(self.n_vars, self.n_vars)).astype(jnp.int32)

        # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
        dag = jnp.tril(mat, k=-1)

        # randomly permute
        key, subk = random.split(key)
        P = random.permutation(subk, jnp.eye(self.n_vars, dtype=jnp.int32))
        dag_perm = P.T @ dag @ P

        if return_mat:
            return dag_perm

        g = mat_to_graph(dag_perm)
        return g

    def unnormalized_log_prob_single(self, *, g, j):
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return n_parents * jnp.log(self.p_edge) + (self.n_vars - n_parents - 1) * jnp.log(1 - self.p_edge)

    def unnormalized_log_prob(self, *, g):
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = len(g.es)

        return E * jnp.log(self.p_edge) + (N - E) * jnp.log(1 - self.p_edge)

    def unnormalized_log_prob_soft(self, *, soft_g):
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = soft_g.sum()
        return E * jnp.log(self.p_edge) + (N - E) * jnp.log(1 - self.p_edge)


class ScaleFreeDAGDistribution(GraphDistribution):
    """
    Randomly-oriented scale-free random graph with power-law degree distribution.
    The pmf is defined as

    :math:`p(G) \\propto \\prod_j (1 + \\text{deg}(j))^{-3}`

    where :math:`\\text{deg}(j)` denotes the in-degree of node :math:`j`

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable

    """

    def sample_G(self, key, return_mat=False):
        pyrandom.seed(int(key.sum()))
        perm = random.permutation(key, self.n_vars).tolist()
        g = ig.Graph.Barabasi(n=self.n_vars, m=self.n_edges_per_node, directed=True).permute_vertices(perm)

        if return_mat:
            return graph_to_mat(g)
        return g

    def unnormalized_log_prob_single(self, *, g, j):
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return -3 * jnp.log(1 + n_parents)

    def unnormalized_log_prob(self, *, g):
        return jnp.array([self.unnormalized_log_prob_single(g=g, j=j) for j in range(self.n_vars)]).sum()

    def unnormalized_log_prob_soft(self, *, soft_g):
        soft_indegree = soft_g.sum(0)
        return jnp.sum(-3 * jnp.log(1 + soft_indegree))


class UniformDAGDistributionRejection(GraphDistribution):
    """
    Naive implementation of a uniform distribution over DAGs via rejection
    sampling. This is efficient up to roughly :math:`d = 5`.
    Properly sampling a uniformly-random DAG is possible but nontrivial
    and not implemented here.

    Args:
        n_vars (int): number of variables in DAG

    """

    def sample_G(self, key, return_mat=False):
        while True:
            key, subk = random.split(key)
            mat = random.bernoulli(subk, p=0.5, shape=(self.n_vars, self.n_vars)).astype(jnp.int32)
            mat = mat.at[..., jnp.arange(self.n_vars), jnp.arange(self.n_vars)].set(0)

            if mat_is_dag(mat):
                if return_mat:
                    return mat
                
                return mat_to_graph(mat)

    def unnormalized_log_prob_single(self, *, g, j):
        return jnp.array(0.0)

    def unnormalized_log_prob(self, *, g):
        return jnp.array(0.0)

    def unnormalized_log_prob_soft(self, *, soft_g):
        return jnp.array(0.0)
