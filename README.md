# Generalized posteriors in ABC

Code for the experiments in [_Generalized Posteriors in Approximate Bayesian Computation_ ](https://arxiv.org/abs/2011.08644).

```
misc{schmon2020generalized,
      title={Generalized Posteriors in Approximate Bayesian Computation}, 
      author={Sebastian M Schmon and Patrick W Cannon and Jeremias Knoblauch},
      year={2020}, eprint={2011.08644}, archivePrefix={arXiv}, primaryClass={stat.ME}
}
```

## Installation
The required dependencies can be installed using `pip install -r requirements.txt`.

## Overview
The code comprises: 

* a collection of kernels `kernels.py`,
* an ABC-MCMC class and algorithm `gb_alg.py`,
* a series of three experiments `gb_experiments.py`, and
* plotting the output of the experiments `gb_plots.py`

## Experiments

Three experiments are described in the paper and can be recreated here (up to stochasticity in the ABC algorithms). 

The first is a straightforward demonstration of the output of the algorithms (the posterior samples) as a function of the weight w chosen. Find the code for this experiment in `initial_experiment` from `gb_experiments.py`, and plotted by `initial_experiment_plot` from `gb_plots.py`.

<img src=img/initial.png height = 250>

The second experiment, detailing the effective sample size of the ABC-MCMC samples is performed by `loss_average_experiment` from `gb_experiments.py`, and plotted by `loss_average_plot` from `gb_plots.py`.

<img src=img/ess_experiment.png width="300" height="300">

Finally the experiment on robustness to misspecification is performed by `misspecification_experiment` in `gb_alg.py`, and plotted by `misspecification_plot` from `gb_plots.py`.

<img src=img/robust_experiment.png width="300" height="300">

