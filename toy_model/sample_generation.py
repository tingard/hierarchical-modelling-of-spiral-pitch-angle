import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
base_t = np.linspace(0, np.pi, 500)


def log_spiral(t, phi, c=0):
    # r = exp(tan(phi) * t + c)
    return np.exp(np.tan(np.deg2rad(phi)) * t + c)


def gen_noisy_arm(phi, N=500, t_offset=None,
                  t_noise_amp=1E-4, r_noise_amp=3E-2):
    t = np.linspace(0, np.pi, N)
    r = log_spiral(t, phi)
    if t_offset is None:
        t_offset = np.random.random() * 2 * np.pi

    t_noise = np.random.normal(loc=0, scale=t_noise_amp, size=t.shape)
    r_noise = np.random.normal(loc=0, scale=r_noise_amp,
                               size=r.shape)

    T = t + t_offset + t_noise
    R = r / r.max() + r_noise
    return T[R > 0.1], R[R > 0.1]


def gen_galaxy(n_arms, pa, sigma_pa, pa_bounds=(0.1, 60), **kwargs):
    translated_bounds = (np.array(pa_bounds) - pa) / sigma_pa
    pas = st.truncnorm.rvs(*translated_bounds, loc=pa, scale=sigma_pa, size=n_arms)
    base_offset = np.random.random() * 2 * np.pi
    return [
        gen_noisy_arm(
            pas[i],
            t_offset=2*np.pi * i / n_arms + base_offset,
            **{'N': 500, **kwargs}
        )
        for i in range(n_arms)
    ]


def fit_log_spiral(t, r):
    def _f(p):
        R = log_spiral(t, p[0])*np.exp(p[1])
        return mean_squared_error(r, R)

    res = minimize(_f, (20, 0))
    if not res['success']:
        return np.nan, np.nan
    return res['x']
