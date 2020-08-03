import os
import sys
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from gzbuilder_analysis.spirals import xy_from_r_theta
import sample_generation as sg
from tqdm import tqdm

loc = os.path.abspath(os.path.dirname(__file__))

n_arms = 20  # np.random.poisson(0.9) + 2
galaxy = sg.gen_galaxy(n_arms, 15, 5, N=250)

# plot the "galaxy" used
for (_t, _r) in galaxy:
    plt.plot(*xy_from_r_theta(_r, _t), '.')
plt.savefig(
    os.path.join(loc, 'plots/one_galaxy.png'),
    bbox_inches='tight'
)

point_data = np.concatenate([
    np.stack((arm_T, arm_R, np.tile(arm_n, len(arm_T))), axis=-1)
    for arm_n, (arm_T, arm_R) in enumerate(galaxy)
])
# point_data_df = pd.DataFrame(point_data, columns=('T', 'R', 'arm_idx'))

T, R, arm_idx = point_data.T
arm_idx = arm_idx.astype(int)


gal_separate_fit_params = pd.Series([])
with tqdm([arm for arm in galaxy]) as bar:
    for i, arm in enumerate(bar):
        gal_separate_fit_params.loc[i] = sg.fit_log_spiral(*arm)

gal_separate_fit_params = gal_separate_fit_params.apply(
    lambda v: pd.Series(v, index=('pa', 'c'))
)
print(gal_separate_fit_params.describe())

with pm.Model() as model:
    # model r = a * exp(tan(phi) * t) as log(r) = b * t + c,
    # but have a uniform prior on phi rather than the gradient!
    gal_pa_mu = pm.TruncatedNormal(
        'pa',
        mu=15, sd=10,
        lower=0, upper=45,
        testval=10,
    )
    gal_pa_sd = pm.HalfCauchy('pa_sd', beta=10, testval=0.1)
    arm_c = pm.Uniform(
        'c',
        lower=-1, upper=0,
        shape=n_arms,
        testval=gal_separate_fit_params['c'].values / 10
    ) * 10
    sigma = pm.HalfCauchy('sigma100', beta=5, testval=1) / 100

    # we want this:
    # arm_pa = pm.TruncatedNormal(
    #     'arm_pa',
    #     mu=gal_pa_mu, sd=gal_pa_sd,
    #     lower=0.1, upper=60,
    #     shape=n_arms,
    # )
    # Specified in a non-centred way:
    arm_pa_mu_offset = pm.Normal(
        'arm_pa_mu_offset',
        mu=0, sd=1, shape=n_arms
    )
    arm_pa = pm.Deterministic(
        'arm_pa',
        gal_pa_mu + gal_pa_sd * arm_pa_mu_offset
    )
    pm.Potential(
        'arm_pa_mu_bound',
        (
            tt.switch(tt.all(arm_pa > 0.1), 0, -np.inf)
            + tt.switch(tt.all(arm_pa < 70), 0, -np.inf)
        )
    )

    arm_b = pm.Deterministic('b', tt.tan(np.pi / 180 * arm_pa))
    arm_r = tt.exp(arm_b[arm_idx] * T + arm_c[arm_idx])
    # prevent r from being very large
    pm.Potential(
        'arm_r_bound',
        tt.switch(tt.all(arm_r < 1E4), 0, -np.inf)
    )
    likelihood = pm.Normal(
        'L',
        mu=arm_r,
        sigma=sigma,
        observed=R
    )

# it's important we now check the model specification, namely do we have any
# problems with logp being undefined?
with model:
    print(model.check_test_point())

with model:
    trace = pm.sample(500, tune=500, target_accept=0.95, init='advi+adapt_diag')

# display the total number and percentage of divergent
divergent = trace['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size / len(trace) * 100
print('Percentage of Divergent %.1f' % divperc)

# Save a traceplot
pm.traceplot(trace, var_names=('pa', 'pa_sd', 'sigma100'))
plt.savefig(
    os.path.join(loc, 'plots/one_galaxy_traceplot.png'),
    bbox_inches='tight'
)
plt.close()

# Save a posterior plot
pm.plot_posterior(trace, var_names=('pa', 'pa_sd', 'sigma100'))
plt.savefig(
    os.path.join(loc, 'plots/one_galaxy_posterior.png'),
    bbox_inches='tight'
)
plt.close()
