import os
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from gzbuilder_analysis.spirals import xy_from_r_theta
from sample_generation import gen_noisy_arm

T, R = gen_noisy_arm(20)

plt.plot(*xy_from_r_theta(R, T))
plt.axis('equal')

with pm.Model() as model:
    # model r = a * exp(tan(phi) * t) as log(r) = b * t + c,
    # but have a uniform prior on phi rather than the gradient!
    pa = pm.Uniform('pa', lower=5, upper=60)

    b = pm.Deterministic('b', tt.tan(np.pi / 180 * pa))
    c = pm.Normal('c', mu=0, sigma=20)

    sigma = pm.HalfCauchy('sigma', beta=1)

    likelihood = pm.Normal(
        'y',
        mu=b * T + c,
        sigma=sigma,
        observed=np.log(R)
    )

    trace = pm.sample(3000, tune=1000, target_accept=0.9)

loc = os.path.abspath(os.path.dirname(__file__))

pm.traceplot(trace, var_names=('pa', 'sigma'))
plt.savefig(
    os.path.join(loc, 'plots/one_arm_traceplot.png'),
    bbox_inches='tight'
)
plt.close()

plt.plot(*xy_from_r_theta(R, T), '.')
plt.savefig(
    os.path.join(loc, 'plots/one_arm.png'),
    bbox_inches='tight'
)
