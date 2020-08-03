import os
import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from gzbuilder_analysis.spirals import xy_from_r_theta
import sample_generation as sg
import argparse
from tqdm import tqdm
from hierarchial_model import BHSM


loc = os.path.abspath(os.path.dirname(__file__))
parser = argparse.ArgumentParser(
    description=(
        'Fit Aggregate model and best individual'
        ' model for a galaxy builder subject'
    )
)
parser.add_argument('--ngals', '-n', metavar='N', default=25,
                    type=int, help='Number of galaxies in sample')
parser.add_argument('--mu', metavar='N', default=20,
                    type=str, help='Global mean pitch angle')
parser.add_argument('--sd', metavar='N', default=5,
                    type=str, help='Inter-galaxy pitch angle std')
parser.add_argument('--sd2', metavar='N', default=10,
                    type=str, help='Intra-galaxy pitch angle std')

args = parser.parse_args()

# Base parameter definition
N_GALS = args.ngals
BASE_PA = args.mu
INTER_GAL_SD = args.sd
INTRA_GAL_SD = args.sd2
N_POINTS = 100
PA_LIMS = (0.1, 60)

print((
    'Making sample of {} galaxies with global mean pa {:.2f}'
    '\nInter-galaxy pitch angle std: {:.2e}'
    '\nIntra-galaxy pitch angle std: {:.2e}'
).format(N_GALS, BASE_PA, INTER_GAL_SD, INTRA_GAL_SD))

gal_pas = st.truncnorm.rvs(*PA_LIMS, loc=BASE_PA, scale=INTER_GAL_SD,
                           size=N_GALS)
gal_n_arms = [np.random.poisson(0.75) + 2 for i in range(N_GALS)]

print('Input galaxies:')
print(pd.DataFrame({
    'Pitch angle': gal_pas,
    'Arm number': gal_n_arms
}).describe())

# Generate our galaxies
galaxies = [
    sg.gen_galaxy(gal_n_arms[i], gal_pas[i], INTRA_GAL_SD, N=N_POINTS)
    for i in range(N_GALS)
]

bhsm = BHSM(galaxies)

# it's important we now check the model specification, namely do we have any
# problems with logp being undefined?
with bhsm.model as model:
    print(model.check_test_point())

# Sampling
with bhsm.model as model:
    db = pm.backends.Text('saved_many_galaxies_trace')
    trace = pm.sample(500, tune=500, target_accept=0.95, max_treedepth=20,
                      init='advi+adapt_diag', trace=db)

    # Save the model
    try:
        pm.model_to_graphviz(model)
        plt.savefig(
            os.path.join(loc, 'plots/many_galaxies_model.png'),
            bbox_inches='tight'
        )
        plt.close()
    except ImportError:
        pass

# Save a traceplot
pm.traceplot(
    trace,
    var_names=('pa', 'pa_sd', 'gal_pa_sd', 'sigma'),
    lines=(
        ('pa_scaled', {}, BASE_PA),
        ('pa_sd', {}, INTER_GAL_SD),
        ('gal_pa_sd', {}, INTRA_GAL_SD),
    )
)
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies_trace.png'),
    bbox_inches='tight'
)
plt.close()

# Save a posterior plot
pm.plot_posterior(
    trace,
    var_names=('pa', 'pa_sd', 'gal_pa_sd', 'sigma')
)
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies_posterior.png'),
    bbox_inches='tight'
)
plt.close()

# plot all the "galaxies" used
s = int(np.ceil(np.sqrt(N_GALS)))
f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(8, 8), dpi=100
)
axs = [j for i in axs_grid for j in i]
for i, ax in enumerate(axs):
    plt.sca(ax)
    try:
        for arm in galaxies[i]:
            plt.plot(*xy_from_r_theta(arm[1], arm[0]))
    except IndexError:
        pass
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies.png'),
    bbox_inches='tight'
)
plt.close()

# make a plot showing arm predictions
with model:
    param_predictions = pm.sample_posterior_predictive(
        trace, samples=50,
        vars=(bhsm.arm_pa, bhsm.arm_c)
    )
pred_pa = param_predictions['arm_pa']
pred_c = param_predictions['c']

f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=100
)
axs = [j for i in axs_grid for j in i]
for j in range(sum(bhsm.gal_n_arms)):
    t = bhsm.T[bhsm.arm_idx == j]
    r = bhsm.R[bhsm.arm_idx == j]
    axs[bhsm.gal_arm_map[j]].plot(
        *xy_from_r_theta(r, t),
        'k.',
        c='C{}'.format(j % 10),
    )
for i in range(len(param_predictions)):
    arm_pa = pred_pa[i]
    arm_c = pred_c[i]
    arm_b = np.tan(np.deg2rad(arm_pa))
    for j in range(len(arm_pa)):
        t = bhsm.T[bhsm.arm_idx == j]
        r_pred = np.exp(arm_b[j] * t + arm_c[j])
        axs[bhsm.gal_arm_map[j]].plot(
            *xy_from_r_theta(r_pred, t),
            c='k',
            alpha=0.5,
            linewidth=1,
        )
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies_predictions.png'),
    bbox_inches='tight'
)
plt.close()

gal_separate_fit_params = pd.Series([])
with tqdm([galaxy for galaxy in galaxies]) as bar:
    for i, gal in enumerate(bar):
        gal_separate_fit_params.loc[i] = [
            sg.fit_log_spiral(*arm)
            for arm in gal
        ]
arm_separate_fit_params = pd.DataFrame(
    [j for _, i in gal_separate_fit_params.items() for j in i],
    columns=('pa', 'c')
)

f, axs_grid = plt.subplots(
    ncols=s, nrows=s,
    sharex=True, sharey=True,
    figsize=(16, 16), dpi=100
)
axs = [j for i in axs_grid for j in i]
for j in range(sum(bhsm.gal_n_arms)):
    t = bhsm.T[bhsm.arm_idx == j]
    r = bhsm.R[bhsm.arm_idx == j]
    axs[bhsm.gal_arm_map[j]].plot(
        *xy_from_r_theta(r, t),
        'k.',
        c='C{}'.format(j % 10),
    )
for i in range(len(param_predictions)):
    arm_pa = pred_pa[i]
    arm_c = pred_c[i]
    arm_b = np.tan(np.deg2rad(arm_pa))
    for j in range(len(arm_pa)):
        t = bhsm.T[bhsm.arm_idx == j]
        r_pred = np.exp(arm_b[j] * t + arm_c[j])
        axs[bhsm.gal_arm_map[j]].plot(
            *xy_from_r_theta(r_pred, t),
            c='g',
            alpha=0.5,
            linewidth=3,
        )

for i, ax in enumerate(axs):
    plt.sca(ax)
    try:
        for p, arm in zip(gal_separate_fit_params.iloc[i], galaxies[i]):
            R_fit = sg.log_spiral(arm[0], p[0])*np.exp(p[1])
            plt.plot(*xy_from_r_theta(R_fit, arm[0]), 'r', alpha=1)

    except IndexError:
        pass
plt.savefig(
    os.path.join(loc, 'plots/many_galaxies_prediction_comparison.png'),
    bbox_inches='tight'
)
plt.close()

print('Trace Summary:')
print(pm.summary(trace).round(2).sort_values(by='Rhat', ascending=False))
