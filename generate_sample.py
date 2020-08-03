import numpy as np
import pandas as pd
import warnings


def generate_sample(n_gals=None, seed=None, pa_filter=lambda a: True,
                    arms_df='lib/merged_arms.pickle'):

    if seed is not None:
        np.random.seed(seed)

    # sample extraction
    galaxies_df = pd.read_pickle(arms_df)

    n_arms = galaxies_df.drop('pipeline', axis=1).apply(lambda a: len(a.dropna()), axis=1)

    # keep only galaxies with one arm or more
    galaxies_df = galaxies_df[n_arms > 0]

    # We want to scale r to have unit variance
    # get all the radial points and calculate their std
    normalization = np.concatenate(
        galaxies_df.drop('pipeline', axis=1)
        .T.unstack().dropna().apply(lambda a: a.R).values
    ).std()
    galaxies = pd.Series([
        [
            np.array((arm.t * arm.chirality, arm.R / normalization))
            for arm in galaxy.dropna()
        ]
        for _, galaxy in galaxies_df.drop('pipeline', axis=1).iterrows()
    ], index=galaxies_df.index)

    # restrict to galaxies with pitch angles between cot(4) and cot(1)
    gal_pas = galaxies_df.apply(
        lambda row: row['pipeline'].get_pitch_angle(row.dropna().values[1:])[0],
        axis=1
    ).reindex_like(galaxies)

    # filter
    mask = gal_pas.apply(pa_filter)
    galaxies = galaxies[mask]

    if n_gals is not None and n_gals > 0 and n_gals < len(galaxies):
        galaxies = galaxies.iloc[
            np.random.choice(
                np.arange(len(galaxies)),
                size=n_gals, replace=False
            )
        ]
        if len(galaxies) < n_gals:
            warnings.warn('Sample contains fewer galaxies than specified')
    return galaxies
