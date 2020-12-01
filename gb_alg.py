import numpy as np
import scipy.stats as stats

from tqdm import tqdm


class Mcmc:
    """ The ABC-MCMC algorithm and its settings."""

    def __init__(self, kernel, is_summary, initial_param, n_particles):
        """Create the Mcmc algorithm.

        Args:
            kernel (Kernel): The kernel used in weighting each model simulation.
            is_summary (bool): Should full data be used, or summary statistics.
            initial_param (float): Initial Markov chain value.
            n_particles (int): The number of model simulations taken at each param value.
        """
        self.kernel = kernel
        self.is_summary = is_summary
        self.initial_param = initial_param
        self.n_particles = n_particles
        
    # Standard deviation of MCMC proposal distribution
    prop_sd = 0.1414
    # Prior distribution params
    prior_mean = 0.0
    prior_sd = 5.0

    def prior_sample(self, n=1):
        """Sample from the prior distribution."""
        return np.random.normal(Mcmc.prior_mean, Mcmc.prior_sd, n)

    def prior_ldensity(self, x):
        """Log-density of the prior distribution."""
        return stats.norm.logpdf(x, Mcmc.prior_mean, Mcmc.prior_sd)

    def generative_model(self, theta, sd, n=100):
        """A single run of the generative model.

        Args:
            theta (float): mean of the model.
            sd (float): standard deviation of the model.
            n (int, optional): The number of iid normal draws defining
            a single run of the model. Defaults to 100.

        Returns:
            np.ndarray: Full output (length n) or summary statistics
            of this full output (length 2).
        """
        out = np.random.normal(theta, sd, n)
        return summary_stats(out) if self.is_summary else out

    def simulator(self, th, sd=1.0):
        """Runs the generative model n_particles times.

        Args:
            th (float): Mean of the generative model.
            sd (float, optional): Standard deviation of the generative model. Defaults to 1.0.

        Returns:
            np.ndarray: n_particles rows of iid draws from gen. model, summarised
            or not according to is_summary.
        """
        dim = 2 if self.is_summary else 100
        out = np.zeros((self.n_particles, dim))
        for i in range(self.n_particles):
            out[i, :] = self.generative_model(th, sd)
        return out

    def run(self, observation, n_mcmc_samples):
        """Runs the ABC-MCMC algorithm itself.

        Args:
            observation (np.ndarray): Full observation from true data generating process.
            n_mcmc_samples (int): The number of iterations of MCMC algorithm.

        Returns:
            (np.ndarray, float): The accepted posterior samples, and the acceptance rate.
        """
        observation = summary_stats(observation) if self.is_summary else observation

        lunif = np.log(np.random.rand(n_mcmc_samples))
        gaussian = np.random.normal(0, 1, n_mcmc_samples)
        acc_samples = np.zeros(n_mcmc_samples)
        acc_samples[0] = self.initial_param
        initial_state_prop = self.simulator(th=self.initial_param)
        prev_lkernel = self.kernel.loss_evaluate(initial_state_prop, observation)
        novel_count = 1

        for i in tqdm(np.arange(1, n_mcmc_samples), leave=False, desc="MCMC algorithm", position=1):

            th_prop = acc_samples[i-1] + 2*Mcmc.prop_sd*gaussian[i-1]
            state_prop = self.simulator(th=th_prop)
            prop_lkernel = self.kernel.loss_evaluate(state_prop, observation)
            lratio = prop_lkernel - prev_lkernel + \
                self.prior_ldensity(th_prop) - self.prior_ldensity(acc_samples[i-1])

            if lunif[i-1] < lratio:
                acc_samples[i] = th_prop
                prev_lkernel = prop_lkernel
                novel_count += 1
            else:
                acc_samples[i] = acc_samples[i-1]

        return acc_samples, novel_count/n_mcmc_samples

    def __repr__(self):
        return str(self.kernel) + "\n" + \
            str(self.observation) + "\n" + str(self.is_summary)


def summary_stats(samples):
    """Applies summary statistics to a sample.

    Args:
        samples (np.ndarray): Samples from a univariate distribution.

    Returns:
        np.ndarray: The mean and variance of the samples.
    """
    return np.array([np.mean(samples), np.var(samples, ddof=1)])


def true_dgp(theta, sd, n=100):
    """Sample from the true data-generating process."""
    np.random.seed(1)
    return np.random.normal(theta, sd, n)
