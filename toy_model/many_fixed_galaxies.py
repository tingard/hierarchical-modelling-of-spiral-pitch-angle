import os
import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from gzbuilder_analysis.spirals import xy_from_r_theta
from sample_generation import gen_galaxy, fit_log_spiral
from tqdm import tqdm

# Base parameter definition
N_GALS = 25
BASE_PA = 15
INTER_GAL_SD = 10
INTRA_GAL_SD = 0

loc = os.path.abspath(os.path.dirname(__file__))

gal_pas = st.truncnorm.rvs(0.1, 60, loc=BASE_PA, scale=INTER_GAL_SD,
                           size=N_GALS)
gal_n_arms = [np.random.poisson(0.75) + 2 for i in range(N_GALS)]

print('Input galaxies:')
print(pd.DataFrame({
    'Pitch angle': gal_pas,
    'Arm number': gal_n_arms
}).describe())

# map from arm to galaxy (so gal_arm_map[5] = 3 means the 5th arm is from the
# 3rd galaxy)
gal_arm_map = np.concatenate([
    np.tile(i, n) for i, n in enumerate(gal_n_arms)
])

# Generate our galaxies
galaxies = [
    gen_galaxy(gal_n_arms[i], gal_pas[i], INTRA_GAL_SD, N=100)
    for i in range(N_GALS)
]

# Create an array containing needed information in a stacked form
point_data = np.concatenate([
    np.stack((
        arm_T, arm_R,
        np.tile(sum(gal_n_arms[:gal_n]) + arm_n, len(arm_T)),
        np.tile(gal_n, len(arm_T))
    ), axis=-1)
    for gal_n, galaxy in enumerate(galaxies)
    for arm_n, (arm_T, arm_R) in enumerate(galaxy)
])

R, T, arm_idx, gal_idx = point_data.T
arm_idx = arm_idx.astype(int)

# ensure the arm indexing makes sense
assert sum(np.unique(arm_idx) - np.arange(sum(gal_n_arms))) == 0

# Define Stochastic variables
with pm.Model() as model:
    # model r = a * exp(tan(phi) * t) + sigma as r = exp(b * t + c) + sigma,
    # and have a uniform prior on phi rather than the gradient!
    # Note our test values should not be the optimum, as then there is little
    # Gradient information to work with! It's about finding a balance between
    # the initial log probability and sufficient gradient for NUTS

    # Global mean pitch angle
    global_pa_mu = pm.Uniform(
        'pa',
        lower=0.01, upper=60,
        testval=20
    )

    # inter-galaxy dispersion
    global_pa_sd = pm.HalfCauchy('pa_sd', beta=10, testval=1)

    # arm offset parameter
    arm_c = pm.Cauchy('c', alpha=0, beta=10, shape=len(gal_arm_map),
                      testval=np.tile(0, len(gal_arm_map)))
    # arm_c = pm.Uniform('c', lower=-10, upper=10, shape=len(gal_arm_map),
    #                    testval=np.tile(0, len(gal_arm_map)))

    # radial noise (degenerate with error on pitch angle I think...)
    sigma = pm.HalfCauchy('sigma', beta=0.1, testval=0.1)


# Define Dependent variables
with model:
    # we want this:
    # gal_pa_mu = pm.TruncatedNormal(
    #     'gal_pa_mu',
    #     mu=global_pa_mu, sd=global_pa_sd,
    #     lower=0.1, upper=60,
    #     shape=n_gals,
    # )
    # arm_pa = pm.TruncatedNormal(
    #     'arm_pa_mu',
    #     mu=gal_pa_mu[gal_arm_map], sd=gal_pa_sd[gal_arm_map],
    #     lower=0.1, upper=60,
    #     shape=len(gal_arm_map),
    # )
    # Specified in a non-centred way:

    gal_pa_mu_offset = pm.Normal(
        'gal_pa_mu_offset',
        mu=0, sd=1, shape=N_GALS,
        testval=np.tile(0, N_GALS)
    )
    gal_pa_mu = pm.Deterministic(
        'gal_pa_mu',
        global_pa_mu + gal_pa_mu_offset * global_pa_sd
    )

    # use a Potential for the truncation, pm.Potential('foo', N) simply adds N
    # to the log likelihood
    pm.Potential(
        'gal_pa_mu_bound',
        (
            tt.switch(tt.all(gal_pa_mu > 0.1), 0, -np.inf)
            + tt.switch(tt.all(gal_pa_mu < 70), 0, -np.inf)
        )
    )

    arm_pa = gal_pa_mu[gal_arm_map]

    # convert to a gradient for a linear fit
    arm_b = pm.Deterministic('b', tt.tan(np.pi / 180 * arm_pa))

    # likelihood function
    likelihood = pm.Normal(
        'Likelihood',
        mu=tt.exp(arm_b[arm_idx] * T + arm_c[arm_idx]),
        sigma=sigma,
        observed=R
    )

# it's important we now check the model specification, namely do we have any
# problems with logp being undefined?
with model:
    print(model.check_test_point())

# Sampling
with model:
    trace = pm.sample(
        1500,
        tune=1000,
        target_accept=0.95,
    )

    # Save the model
    try:
        pm.model_to_graphviz(model)
        plt.savefig(
            os.path.join(loc, 'plots/many_fixed_galaxies_model.png'),
            bbox_inches='tight'
        )
        plt.close()
    except ImportError:
        pass


# Save a traceplot
pm.traceplot(
    trace,
    var_names=('pa', 'pa_sd', 'sigma'),
    combined=True,
    lines=(
        ('pa', {}, BASE_PA),
        ('pa_sd', {}, INTER_GAL_SD),
        ('gal_pa_sd', {}, INTRA_GAL_SD),
    )
)
plt.savefig(
    os.path.join(loc, 'plots/many_fixed_galaxies_trace.png'),
    bbox_inches='tight'
)
plt.close()

# Save a posterior plot
pm.plot_posterior(trace, var_names=('pa', 'pa_sd', 'sigma'))
plt.savefig(
    os.path.join(loc, 'plots/many_fixed_galaxies_posterior.png'),
    bbox_inches='tight'
)
plt.close()

# plot all the "galaxies" used
s = int(np.ceil(np.sqrt(N_GALS)))
f, axs_grid = plt.subplots(ncols=s, nrows=s, figsize=(8, 8), dpi=100)
axs = [j for i in axs_grid for j in i]
for i, ax in enumerate(axs):
    plt.sca(ax)
    try:
        for arm in galaxies[i]:
            plt.plot(*xy_from_r_theta(arm[1], arm[0]))
        plt.axis('equal')
    except IndexError:
        pass
plt.savefig(
    os.path.join(loc, 'plots/many_fixed_galaxies.png'),
    bbox_inches='tight'
)
