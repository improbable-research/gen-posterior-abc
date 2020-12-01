import numpy as np
import scipy.stats as stats

from numba import njit

class SoftKernel:
    """Soft-threshold kernel using Euclidean norm."""
    def __init__(self, weight):      
        self.weight = weight

    def loss_evaluate(self, x, obs):
        """Evaluate the soft-threshold kernel.

        Args:
            x (np.ndarray): [m, d] array - m iid draws from the model.
            obs (np.ndarray): [d,] array - a model observation.

        Returns:
            float: the mean of the losses on each row, with a multiplicative weight.
        """
        return - self.weight * np.mean(np.linalg.norm(x - obs, axis=1))


class K2Kernel:
    """Maximum mean discrepancy loss."""
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def loss_evaluate(self, x, obs):
        """Evaluate the K2-ABC kernel.

        Args:
            x (np.ndarray): Mxd array of M iid draws from the model.
            obs (np.ndarray): length d model observation.

        Returns:
            float: the mean of the losses on each row, with a multiplicative weight.
        """
        losses = np.zeros([x.shape[0]])
        for i, x_row in enumerate(x):
            losses[i] = mmd_est(x_row, obs)
        return -np.mean(losses)/self.epsilon


class WassKernel:
    """Wasserstein distance loss."""
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def loss_evaluate(self, x, obs):
        """Evaluate the Wasserstein distance kernel.

        Args:
            x (np.ndarray): Mxd array of M iid draws from the model.
            obs (np.ndarray): length d model observation.

        Returns:
            float: the mean of the losses on each row, with a multiplicative weight.
        """
        losses = np.zeros([x.shape[0]])
        for i, x_row in enumerate(x):
            losses[i] = stats.wasserstein_distance(x_row, obs)
        return -np.mean(losses)/self.epsilon


@njit
def gauss_rbf(xi, xj, c=1/5):
    """Gaussian radial basis function (rbf).

    Args:
        xi (float): i'th element of np.ndarray x.
        xj (float): j'th element of np.ndarray x.
        c (float, optional): Scaling factor in definition of rbf. Defaults to 1/5.

    Returns:
        float: evaluation of rbf.
    """
    return np.exp(-c*(xi-xj)**2)


@njit
def mmd_est(x, y):
    """Produces an unbiased estimate of the maximum mean discprepancy (mmd) 
    between two distributions using samples from each. See p.4 [1].

    [1] K2-ABC: Approximate Bayesian Computation with Kernel Embeddings.
        Park, M., Jitkrittum, W., Sejdinovic, D. 2015

    Args:
        x (np.ndarray): 1d samples from first distribution.
        y (np.ndarray): 1d samples from second distribution.

    Returns:
        float: The mmd estimate.
    """
    n_x = len(x)
    n_y = len(y)

    factor1 = 0
    for i in range(n_x):
        for j in range(n_x):
            if (j == i): continue
            factor1 += gauss_rbf(x[i], x[j])
    factor1 /= (n_x*(n_x-1))

    factor2 = 0
    for i in range(n_y):
        for j in range(n_y):
            if (j == i): continue
            factor2 += gauss_rbf(y[i], y[j])
    factor2 /= (n_y*(n_y-1))

    factor3 = 0
    for i in range(n_x):
        for j in range(n_y):
            factor3 += gauss_rbf(x[i], y[j])
    factor3 *= 2/(n_x*n_y)

    return factor1 + factor2 - factor3