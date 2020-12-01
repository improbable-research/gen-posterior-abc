import numpy as np
import pickle

from gb_alg import Mcmc, true_dgp
from kernels import SoftKernel, K2Kernel, WassKernel
from tqdm import tqdm


def initial_experiment():
    """Initial tests - posterior sample densities against true and misspecified posteriors."""
    # True DGP 
    dgp_th = 1.0
    dgp_sig = np.sqrt(2.0)    # Standard case
    # dgp_sig = np.sqrt(0.5)      # Alternative
    obs = true_dgp(theta=dgp_th, sd=dgp_sig)

    # Run MCMC-ABC 
    summary = [True, False, False]
    weight_schedule = [5, 24, 56]
    particle_schedule = [32, 2, 2]
    sample_store = [[] for i in range(3)]
    acc_store = [[] for i in range(3)]

    for w in tqdm(weight_schedule, desc="Experiment progress"):

        kernels = [SoftKernel(weight=w), K2Kernel(epsilon=1/(8*w)), WassKernel(epsilon=1/w)]

        for j, kernel in enumerate(kernels):

            mcmc = Mcmc(
                kernel=kernel, is_summary=summary[j], initial_param=1.0, n_particles=particle_schedule[j])
            out, acc_rate = mcmc.run(observation=obs, n_mcmc_samples=10_000)

            sample_store[j].append(out)
            acc_store[j].append(acc_rate)

    # Save data
    out = {'samples': sample_store, 'acc_rates': acc_store,
           'weight_schedule': weight_schedule, 'dgp_th': dgp_th,
           'dgp_sig': dgp_sig, 'observation': obs}

    pickle.dump(out, open("experiments/initial_test/out.p", "wb"))


def loss_average_experiment():
    """Fix the number of calls to the simulator, and split those 
    calls between the number of MCMC steps and the number of 'particles'
    averaged at each step. 
    """
    # True DGP
    dgp_th = 1.0
    dgp_sig = np.sqrt(2.0)    # Standard case
    # dgp_sig = np.sqrt(0.5)      # Alternative
    obs = true_dgp(theta=dgp_th, sd=dgp_sig)

    # Run experiments
    n_sim_calls = 1_500_000
    particle_schedule = np.array([1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60])
    mcmc_schedule = np.round(n_sim_calls/particle_schedule)
    summary = [True, False, False]

    sample_store = [[] for i in range(3)]
    acc_store = [[] for i in range(3)]
    kernels = [SoftKernel(weight=35.0), K2Kernel(epsilon=0.005), WassKernel(epsilon=0.05)]

    for i, _ in enumerate(tqdm(particle_schedule, desc="Experiment progress")):
        for j, kernel in enumerate(kernels):

            mcmc = Mcmc(kernel=kernel, is_summary=summary[j], initial_param=1.0, n_particles=particle_schedule[i])
            out, acc_rate = mcmc.run(observation=obs, n_mcmc_samples=int(mcmc_schedule[i]))
            sample_store[j].append(out)
            acc_store[j].append(acc_rate)

    # Save data
    out = {'samples': sample_store, 'acc_rates': acc_store,
           'particle_schedule': particle_schedule, 'mcmc_schedule': mcmc_schedule, 
           'dgp_th': dgp_th, 'dgp_sig': dgp_sig, 'observation': obs}
    pickle.dump(out, open("experiments/loss_averages/out.p", "wb"))


def misspecification_experiment():
    """Finding the posterior mean under several levels of misspecification, by 
    adjusting the variance of the data generating process.
    """
    # True DGP
    dgp_th = 1.0
    dgp_var = np.linspace(0.5, 5, 10)
    dgp_sig = dgp_var**0.5

    n_mcmc_samples = 3_000_000
    mean_store = [[] for i in range(3)]
    mean_store_64 = [[] for i in range(3)]
    observations = []

    kernels = [SoftKernel(weight=35.0), K2Kernel(epsilon=0.005), WassKernel(epsilon=0.05)]
    summary = [True, False, False]
    low_particle_schedule = [2, 1, 1]

    for i, _ in tqdm(enumerate(dgp_sig), desc="Experiment progress", total=len(dgp_sig)):
        for j, kernel in enumerate(kernels):

            obs = true_dgp(theta=dgp_th, sd=dgp_sig[i])
            observations.append(obs)

            # low_particle_schedule particles
            mcmc = Mcmc(kernel=kernel, is_summary=summary[j], initial_param=1.0, n_particles=low_particle_schedule[j])
            out, _ = mcmc.run(observation=obs, n_mcmc_samples=n_mcmc_samples)
            mean_store[j].append(np.mean(out))

            # 64 particles
            mcmc = Mcmc(kernel=kernel, is_summary=summary[j], initial_param=1.0, n_particles=64)
            out, _ = mcmc.run(observation=obs, n_mcmc_samples=int(n_mcmc_samples/64))
            mean_store_64[j].append(np.mean(out))


    # Save data
    out = {'mean_store': mean_store, 'mean_store_64': mean_store_64, 
            'dgp_th': dgp_th, 'dgp_var': dgp_var, 'observation': observations}
    pickle.dump(out, open("experiments/misspecification/out.p", "wb"))


if __name__ == '__main__':
    # initial_experiment()
    # loss_average_experiment()
    misspecification_experiment()

