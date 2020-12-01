import scipy.stats as stats
import numpy as np
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 


def normal_posterior(prior_mean, prior_sd, lik_sd, observation):
    """Return the posterior for the mean of a normal distribution, 
    for conjugate normal-normal prior-likelihood.

    Args:
        prior_mean (float): Mean of the prior distribution.
        prior_sd (float): Standard deviation of the prior distribution.
        lik_sd (float): Standard deviation of the likelihood.
        observation (np.ndarray): Real observation. 

    Returns:
        Scipy.stats.rv_continous: The posterior distribution of the mean.
    """
    n = len(observation)
    post_var = ((prior_sd**-2) + n * (lik_sd**-2))**-1
    post_mean = (prior_mean * (prior_sd**-2) + sum(observation) * (lik_sd**-2)) * post_var
    return stats.norm(post_mean, np.sqrt(post_var))


def initial_experiment_plot():
    """Plot posterior sample densities against true and misspecified posteriors."""
    # Load experiments 
    out = pickle.load(open("experiments/initial_test/out.p", "rb"))
    samples = out['samples']
    weight_schedule = out['weight_schedule']

    # True DGP 
    dgp_sig = out['dgp_sig']
    obs = out['observation']

    # True DGP posterior (of mean)
    posterior = normal_posterior(prior_mean = 0, prior_sd = 5, lik_sd = dgp_sig, observation=obs)
    
    # Model posterior (of mean)
    mis_sd = 1.0
    mis_posterior = normal_posterior(prior_mean = 0, prior_sd = 5, lik_sd = mis_sd, observation=obs)

    # Paper plots 
    sns.set_context('talk')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1.2]})

    pal = 'rocket'
    color_pal = sns.color_palette(pal, as_cmap=True)
    norm = matplotlib.colors.Normalize(0, 70)
    pcm = matplotlib.cm.ScalarMappable(norm=norm, cmap=color_pal)

    titles = ['ST-ABC', 'K2-ABC', 'W-ABC']
    for k, ax in enumerate([ax1, ax2, ax3]):
        for i, weight in reversed(list(enumerate(weight_schedule))):
            ax = sns.kdeplot(x=samples[k][i], bw_adjust=2.0, linestyle="-", linewidth = 4, ax=ax, color=pcm.to_rgba(weight), fill=True, alpha=0.3)
            ax.set_title(titles[k])

    inputs = np.linspace(posterior.ppf(0.0001), posterior.ppf(0.9999), 500)
    for ax in [ax1, ax2, ax3]:
        ax = sns.lineplot(x=inputs, y=posterior.pdf(inputs), color='black', linewidth = 2, ax=ax, label="True posterior")
        ax = sns.lineplot(x=inputs, y=mis_posterior.pdf(inputs), color='black', linestyle='--', linewidth = 2, ax=ax, label="Mis. posterior")
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlim([0.4, 1.8])
        ax.set_ylim([0, 5])
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('')
        ax.legend(frameon=False, handlelength=1, loc='upper left')

    plt.colorbar(pcm, label="Weight")
    fig.set_size_inches(22.8, 7)
    fig.savefig('initial.png', bbox_inches='tight')


def loss_average_plot():
    """Plot effective sample size as a function of n_particles."""
    # Load experiments 
    out = pickle.load(open("experiments/loss_averages/out.p", "rb"))
    samples = out['samples']
    particle_schedule = out['particle_schedule']

    # ESS 
    ess = [[] for i in range(3)]
    for i in range(len(samples)):
        for run in samples[i]:
            ess[i].append(az.ess(run))

    # Paper plot 
    sns.set_context('talk')
    fig, (ax) = plt.subplots(1, 1)

    labels = ['ST-ABC', 'K2-ABC', 'W-ABC']
    line_cols = ['seagreen', 'royalblue', 'darkorange']

    for i in range(len(samples)):
        ax = sns.lineplot(x=particle_schedule, y=ess[i], ax=ax, linewidth=4,
                        linestyle='--', label=labels[i], color=line_cols[i])

    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend(frameon=False, handlelength=1.8)
    ax.set(xlabel='Number of particles')
    ax.set_ylabel('Effective sample size')
    ax.set(yscale="log")
    fig.set_size_inches(7, 7)
    fig.savefig('loss_average.png', bbox_inches='tight')


def misspecification_plot():
    """Plot posterior mean as a function of level of misspecification."""
    # Load experiments 
    out = pickle.load(open("experiments/misspecification/out.p", "rb"))
    mean_store = out['mean_store']
    mean_store_64 = out['mean_store_64']
    dgp_var = out['dgp_var']

    # Plots
    sns.set_context('talk')
    fig, (ax) = plt.subplots(1, 1)

    line_cols = ['seagreen', 'royalblue', 'darkorange']
    labels = [str('ST-ABC  '+r'$M=2$'), str('K2-ABC  '+r'$M=1$'), str('W-ABC   '+r'$M=1$')]
    labels_64 = [str('ST-ABC  '+r'$M=64$'), str('K2-ABC  '+r'$M=64$'), str('W-ABC   '+r'$M=64$')]

    for i in range(len(mean_store)):
        ax = sns.lineplot(x=dgp_var, y=mean_store[i], linewidth=4.0, linestyle='--',
                        color=line_cols[i], ax=ax, label=labels[i])

    for i in range(len(mean_store_64)):
        ax = sns.lineplot(x=dgp_var, y=mean_store_64[i], linewidth=4.0, linestyle='-.',
                        color=line_cols[i], ax=ax, label=labels_64[i])

    ax.set_ylim([1.0, 1.4])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend(frameon=False, handlelength=1.8)
    fig.set_size_inches(7, 7)
    ax.set_ylabel('Posterior mean')
    ax.set_xlabel(r'$\sigma^2$')
    fig.savefig('misspecification.png', bbox_inches='tight')


def main():
    # initial_experiment_plot()
    # loss_average_plot()
    misspecification_plot()

if __name__ == '__main__':
    main()
