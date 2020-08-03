import argparse
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from gzbuilder_analysis import load_aggregation_results, load_fit_results


parser = argparse.ArgumentParser(
    description=(
        'Run hierarchial model and save output'
    )
)
parser.add_argument('--aggregation-results',
                    default='lib/aggregation_results', type=str,
                    help='Location of Aggregation results file')
parser.add_argument('--tuning-results',
                    default='lib/tuning_results', type=str,
                    help='Location of Tuning results file')

parser.add_argument('--output',
                    default='lib/spiral_arms.pickle', type=str,
                    help='Location of Tuning results file')


args = parser.parse_args()

agg_results = load_aggregation_results(args.aggregation_results)
fit_results = load_fit_results(args.tuning_results)

arms_df = pd.Series([], dtype=object)
with tqdm(agg_results.index, desc='Correcting disks') as bar:
    for subject_id in bar:
        spirals = agg_results.loc[subject_id].spiral_arms
        arms = pd.Series([], dtype=object)
        if subject_id in fit_results.index:
            fit_disk = fit_results.loc[subject_id]['fit_model'].disk
        else:
            fit_disk = None
        for i in range(len(spirals)):
            arm = deepcopy(spirals[i])
            if fit_disk is not None:
                arm.modify_disk(
                    centre=fit_disk[['mux', 'muy']],
                    phi=fit_disk.roll,
                    ba=fit_disk.q,
                )
            arms[i] = arm
        try:
            arms['pipeline'] = arms.loc[0].get_parent()
        except KeyError:
            arms['pipeline'] = None
        arms_df[subject_id] = arms

arms_df.apply(pd.Series).to_pickle(args.output)
