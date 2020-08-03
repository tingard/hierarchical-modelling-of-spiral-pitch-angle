import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt


class BHSM():
    """This is a base model which provides standardized data access for the
    model-specific subclasses below.
    """

    def __init__(self, galaxies, build=True):
        """Accepts a list of groups of arm polar coordinates, and builds a
        PYMC3 hierarchial model to infer global distributions of pitch angle
        """
        self.galaxies = galaxies
        self.gal_n_arms = galaxies.apply(len)
        self.n_arms = self.gal_n_arms.sum()
        # map from arm to galaxy (so gal_arm_map[5] = 3 means the 5th arm is
        # from the 3rd galaxy)
        self.gal_arm_map = np.concatenate([
            np.tile(i, n) for i, n in enumerate(self.gal_n_arms.values)
        ])
        # Create an array containing needed information in a stacked form
        self.data = pd.DataFrame(
            np.concatenate([
                np.stack((
                    arm_T,
                    arm_R,
                    np.tile(sum(self.gal_n_arms.iloc[:gal_n]) + arm_n, len(arm_T)),
                    np.tile(gal_n, len(arm_T))
                ), axis=-1)
                for gal_n, galaxy in enumerate(galaxies)
                for arm_n, (arm_T, arm_R) in enumerate(galaxy)
            ]),
            columns=('theta', 'r', 'arm_index', 'galaxy_index')
        )
        # ensure correct dtypes
        self.data[['arm_index', 'galaxy_index']] = \
            self.data[['arm_index', 'galaxy_index']].astype(int)

        self.point_arm_map = self.data['arm_index'].values
        # assert we do not have any NaNs
        if np.any(self.data.isna()):
            raise ValueError('NaNs present in arm values')

        # ensure the arm indexing makes sense
        indexing_sensible = np.all(
            (
                np.unique(self.data['arm_index'])
                - np.arange(self.n_arms)
            ) == 0
        )
        assert indexing_sensible, 'Something went wrong with arm indexing'

        if build is True:
            self.build_model()
        else:
            self.model = None

    def build_model(self, name=''):
        """Function to be overwritten by specific subclass of BHSM
        """
        pass

    def do_inference(self, draws=1000, tune=500, target_accept=0.85,
                     max_treedepth=20, init='advi+adapt_diag',
                     **kwargs):
        if self.model is None:
            self.build_model()

        # it's important we now check the model specification, namely do we
        # have any problems with logp being undefined?
        with self.model as model:
            print(model.check_test_point())

        # Sampling
        with self.model as model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                max_treedepth=20,
                init=init,
                **kwargs
            )
        return trace

    def do_advi(self):
        raise NotImplementedError('ADVI is not implemented')

    def save(self, output, trace=None):
        if trace is not None:
            trace_fname = pm.save_trace(trace)
        else:
            trace_fname = None

        with open(output, "wb") as buff:
            pickle.dump(
                {
                    'galaxies': self.galaxies,
                    'trace': trace_fname,
                    'n_chains': trace.nchains if trace is not None else None,
                },
                buff
            )

    @classmethod
    def load(cls, input_file):
        saved_result = pd.read_pickle(input_file)
        trace_fname = saved_result.pop('trace', None)

        bhsm = cls(saved_result['galaxies'])
        if trace_fname is not None:
            with bhsm.model:
                trace = pm.load_trace(trace_fname)
        else:
            trace = None
        return {
            **saved_result,
            'bhsm': bhsm,
            'trace': trace
        }


class UniformBHSM(BHSM):
    """This is the model described in the paper Lingard et al. (2020b),
    "Galaxy Zoo Builder: Morphological Dependence of Spiral Galaxy Pitch
    Angle".
    """

    def build_model(self, name=''):
        # Define Stochastic variables
        with pm.Model(name=name) as self.model:
            # Global mean pitch angle
            self.phi_gal = pm.Uniform(
                'phi_gal',
                lower=0, upper=90,
                shape=len(self.galaxies)
            )
            # note we don't model inter-galaxy dispersion here
            # intra-galaxy dispersion
            self.sigma_gal = pm.InverseGamma(
                'sigma_gal',
                alpha=2, beta=20, testval=5
            )
            # arm offset parameter
            self.c = pm.Cauchy(
                'c',
                alpha=0, beta=10,
                shape=self.n_arms,
                testval=np.tile(0, self.n_arms)
            )

            # radial noise
            self.sigma_r = pm.InverseGamma('sigma_r', alpha=2, beta=0.5)

            # ----- Define Dependent variables -----

            # Phi arm is drawn from a truncated normal centred on phi_gal with
            # spread sigma_gal
            self.phi_arm = pm.TruncatedNormal(
                'phi_arm',
                mu=self.phi_gal[self.gal_arm_map], sd=self.sigma_gal,
                lower=0, upper=90,
                shape=self.n_arms
            )

            # transform to gradient for fitting
            self.b = tt.tan(np.pi / 180 * self.phi_arm)

            # r = exp(theta * tan(phi) + c)
            # do not track this as it uses a lot of memory
            r = tt.exp(
                self.b[self.data['arm_index'].values] * self.data['theta']
                + self.c[self.data['arm_index'].values]
            )

            # likelihood function (assume likelihood here)
            self.likelihood = pm.Normal(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                observed=self.data['r'],
            )


class RobustUniformBHSM(BHSM):
    r"""This model uses a Student-t distribution likelihood, so as to be more
    robust to errors.

    THIS IS A WORK IN PROGRESS, the model is not currently converging
    """

    def build_model(self, name=''):
        # Define Stochastic variables
        with pm.Model(name=name) as self.model:
            # Global mean pitch angle
            self.phi_gal = pm.Uniform(
                'phi_gal',
                lower=0, upper=90,
                shape=len(self.galaxies)
            )
            # note we don't model inter-galaxy dispersion here
            # intra-galaxy dispersion
            self.sigma_gal = pm.InverseGamma(
                'sigma_gal',
                alpha=2, beta=20, testval=5
            )
            # arm offset parameter
            self.c = pm.Cauchy(
                'c',
                alpha=0, beta=10,
                shape=self.n_arms,
                testval=np.tile(0, self.n_arms)
            )

            # radial noise
            self.sigma_r = pm.InverseGamma('sigma_r', alpha=2, beta=0.5)

            # define prior for Student T degrees of freedom
            # self.nu = pm.Uniform('nu', lower=1, upper=100)

            # Define Dependent variables
            self.phi_arm = pm.TruncatedNormal(
                'phi_arm',
                mu=self.phi_gal[self.gal_arm_map], sd=self.sigma_gal,
                lower=0, upper=90,
                shape=self.n_arms
            )

            # convert to a gradient for a linear fit
            self.b = tt.tan(np.pi / 180 * self.phi_arm)
            r = tt.exp(
                self.b[self.data['arm_index'].values] * self.data['theta']
                + self.c[self.data['arm_index'].values]
            )

            # likelihood function
            self.likelihood = pm.StudentT(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                nu=1, #self.nu,
                observed=self.data['r'],
            )


class ArchimedianBHSM(BHSM):
    r"""This model fits archimedian spirals, rather than the logarithmic spirals
    present in other models. There is no assumed relationship between arms.

    An Archimedian spiral is given by

    $$r = a\theta^{1/n}$$

    Here we constrain $n$ to be the same for all arms, and add a rotation
    paramter $\delta\theta$

    $$r = a\left(\theta + \delta\theta\right)^{m}$$

    THIS IS A WORK IN PROGRESS, the model is not currently converging
    """

    def build_model(self, n=None, name='archimedian_model'):
        with pm.Model(name=name) as self.model:
            if n is None:
                # one n per galaxy, or per arm?
                self.n_choice = pm.Categorical(
                    'n_choice',
                    [1, 1, 0, 1, 1],
                    testval=1,
                    shape=len(self.galaxies)
                )
                self.n = pm.Deterministic('n', self.n_choice - 2)
                self.chirality_correction = tt.switch(self.n < 0, -1, 1)
            else:
                msg = 'Parameter $n$ must be a nonzero float'
                try:
                    n = float(n)
                except ValueError:
                    pass
                finally:
                    assert isinstance(n, float) and n != 0, msg

                self.n_choice = None
                self.n = pm.Deterministic('n', np.repeat(n, len(self.galaxies)))

            self.chirality_correction = tt.switch(self.n < 0, -1, 1)
            self.a = pm.HalfCauchy(
                'a',
                beta=1, testval=1,
                shape=self.n_arms
            )
            self.psi = pm.Normal(
                'psi',
                mu=0, sigma=1, testval=0.1,
                shape=self.n_arms,
            )
            self.sigma_r = pm.InverseGamma(
                'sigma_r',
                alpha=2, beta=0.5
            )
            # Unfortunately, as we need to reverse the theta points for arms
            # with n < 1, and rotate all arms to start at theta = 0,
            # we need to do some model-mangling
            self.t_mins = pd.Series({
                i: self.data.query('arm_index == @i')['theta'].min()
                for i in np.unique(self.data['arm_index'])
            })
            r_stack = [
                self.a[i]
                * tt.power(
                    (
                        self.data.query('arm_index == @i')['theta'].values
                        - self.t_mins[i]
                        + self.psi[i]
                    ),
                    1 / self.n[int(self.gal_arm_map[i])]
                )[::self.chirality_correction[int(self.gal_arm_map[i])]]
                for i in np.unique(self.data['arm_index'])
            ]
            r = tt.concatenate(r_stack)
            self.likelihood = pm.StudentT(
                'Likelihood',
                mu=r,
                sigma=self.sigma_r,
                observed=self.data['r'].values,
            )

    def do_inference(self, draws=20000, tune=2000, init='adapt_diag', **kwargs):
        if self.model is None:
            self.build_model()

        # it's important we now check the model specification, namely do we
        # have any problems with logp being undefined?
        with self.model as model:
            test_point = model.check_test_point()

            if len(self.model.name) > 0:
                l_key = self.model.name + '_'
            else:
                l_key = ''

        print(test_point)
        if np.isnan(test_point['{}Likelihood'.format(l_key)]):
            print('The model\'s test point had an undefined likelihood, meaning sampling will fail')
            sys.exit(0)

        # Sampling
        with self.model as model:
            if self.n_choice is not None:
                step1 = pm.CategoricalGibbsMetropolis(self.n_choice)
                step2 = pm.Metropolis([self.a, self.psi, self.sigma_r])
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    init=init,
                    step=[step1, step2],
                    **kwargs
                )
            else:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    init=init,
                    **kwargs
                )
        return trace


def get_gal_pas(trace, galaxies, gal_arm_map, name=''):
    if type(galaxies) is not pd.Series:
        galaxies = pd.Series(galaxies)
    if name == '':
        arm_pas = trace['phi_arm']
    else:
        arm_pas = trace[f'{name}_phi_arm']
    gal_pas = pd.Series(index=galaxies.index, dtype=object)
    for i, j in enumerate(gal_pas.index):
        arm_mask = gal_arm_map == j
        weights = list(map(lambda l: l.shape[1], galaxies.loc[j]))
        assert len(weights) == sum(arm_mask.astype(int))
        gal_pas.loc[j] = np.average(
            arm_pas.T[arm_mask],
            weights=weights,
            axis=0,
        )
    gal_pas = gal_pas.apply(pd.Series)
    gal_pas.columns = gal_pas.columns.rename('sample')
    gal_pas.index = gal_pas.index.rename('galaxy')
    return gal_pas


def plot_galaxy(
    sid,
    galaxies=None,
    bhsm=None,
    trace=None,
    many_colored=False,
    markersize=1,
    r_err_alpha=0,
    phi_err_alpha=0.75,
    rasterize_scatter=False,
    plot_fit=True,
):
    assert bhsm is not None or galaxies is not None, \
        "One of `bhsm`, `galaxies` required"
    gals = galaxies if galaxies is not None else bhsm.galaxies
    for i, arm in enumerate(gals.loc[sid]):
        plt.scatter(
            *arm,
            c=(f'C{i}' if many_colored else 'k'),
            s=markersize,
            alpha=0.4,
            rasterized=rasterize_scatter
        )
    if not plot_fit:
        return
    if bhsm is None or trace is None:
        assert (bhsm is not None) and (trace is not None), \
            'Both the BHSM instance and a trace is requred to plot the fit'
    try:
        idx = np.where(bhsm.galaxies.index.values == sid)[0][0]
        phi_arm_samples = trace.phi_arm[:, bhsm.gal_arm_map == idx].T
        c_samples = trace.c[:, bhsm.gal_arm_map == idx].T
        r_err = trace.sigma_r.mean()
        for i, arm in enumerate(bhsm.galaxies.loc[sid]):
            t_pred = np.expand_dims(
                np.linspace(arm[0].min(), arm[0].max(), 200),
                -1
            )
            r_pred = np.exp(
                np.tan(np.deg2rad(phi_arm_samples[i])) * t_pred
                + c_samples[i]
            )
            r_mn = r_pred.mean(axis=1)
            r_sd = r_pred.std(axis=1)
            if r_err_alpha > 0:
                plt.fill_between(
                    t_pred[:, 0],
                    r_mn - r_err,
                    r_mn + r_err,
                    color='r',
                    alpha=r_err_alpha
                )
            plt.fill_between(
                t_pred[:, 0],
                r_mn - 2 * r_sd,
                r_mn + 2 * r_sd,
                color='r',
                alpha=phi_err_alpha
            )
    except ValueError:
        pass
