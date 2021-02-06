import argparse
from time import sleep
import numpy as np
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description=(
        'Merge spiral arms from validation and main run galaxies'
    )
)
parser.add_argument(
    '--spiral-arms',
    default='lib/spiral_arms.pickle', type=str,
    help=(
        'Location of spiral arms file produced by '
        '`extract_spirals_from_gzb_results.py`'
    )
)
parser.add_argument(
    '--duplicate-ids',
    default='lib/duplicate_galaxies.csv', type=str,
    help='Location of CSV containing subject ids of duplicate galaxies'
)

parser.add_argument(
    '--output',
    default='lib/merged_arms.pickle', type=str,
    help='Location of Tuning results file'
)

args = parser.parse_args()
arms = pd.read_pickle(args.spiral_arms)
duplicates = pd.read_csv(args.duplicate_ids, index_col=0)
duplicates = duplicates.rename(
    columns={'0': 'original', '1': 'validation'}
).astype(int)

original_arms = arms.loc[duplicates['original'].values]
validation_arms = arms.loc[duplicates['validation'].values]\
    .drop(columns='pipeline')\
    .rename(columns=lambda c: f'val-{c}')
validation_arms.index = original_arms.index

combined_arms = pd.concat((original_arms, validation_arms), axis=1)
in_val = np.logical_or(
    np.isin(arms.index, duplicates['original']),
    np.isin(arms.index, duplicates['validation'])
)
merged_arms = pd.Series([], dtype=object)
with tqdm(combined_arms.index.values) as bar:
    for idx in bar:
        if combined_arms.loc[idx]['pipeline'] is not None:
            res = combined_arms.loc[idx]['pipeline'].merge_arms(
                combined_arms.loc[idx].drop('pipeline').dropna().values
            )
            merged_arms.loc[idx] = dict(
                pipeline=combined_arms.loc[idx]['pipeline'],
                **{f'arm_{i}': j for i, j in enumerate(res)}
            )
        else:
            sleep(0.1)

final_arms = pd.concat((
    merged_arms.apply(pd.Series),
    arms[~in_val],
), axis=0)

final_arms.to_pickle(args.output)
