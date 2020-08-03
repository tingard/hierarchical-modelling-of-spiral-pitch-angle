import sys
import pickle
import pandas as pd
import pymc3 as pm
from hierarchial_model import UniformBHSM
import generate_sample


import argparse

parser = argparse.ArgumentParser(
    description=(
        'Run hierarchial model and save output'
    )
)
parser.add_argument('--ngals', '-N', default=None, type=int,
                    help='Number of galaxies in sample')
parser.add_argument('--ntune', default=500, type=int,
                    help='Number of tuning steps to take')
parser.add_argument('--ndraws', default=1000, type=int,
                    help='Number of posterior draws to take')
parser.add_argument('--arms-df', metavar='/path/to/file.pickle',
                    default='lib/merged_arms.pickle',
                    help='Where to save output dump')
parser.add_argument('--output', '-o', metavar='/path/to/file.pickle',
                    default='',
                    help='Where to save output dump')

args = parser.parse_args()

# generate a sample using the helper function
galaxies = generate_sample.generate_sample(
    n_gals=args.ngals,
    seed=0,
    arms_df=args.arms_df
)

# define our output file name
if args.output == '':
    args.output = 'n{}d{}t{}.pickle'.format(
        len(galaxies),
        args.ndraws,
        args.ntune,
    )

# initialize the model using the custom BHSM class
bhsm = UniformBHSM(galaxies)

# pm.model_to_graphviz(bhsm.model).view('plots/{}_model_graphviz.pdf')

trace = bhsm.do_inference(
    draws=args.ndraws,
    tune=args.ntune,
)

bhsm.save(args.output, trace)


divergent = trace['diverging']

print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size / len(trace) * 100
print('Percentage of Divergent %.1f' % divperc)

# print('Trace Summary:')
# print(pm.summary(trace).round(2).sort_values(by='r_hat', ascending=False))
