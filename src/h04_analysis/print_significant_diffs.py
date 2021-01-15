import os
import sys
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.append('./src/')
from util import argparser
from util import util


def get_args():
    # Models
    # argparser.add_argument('--checkpoints-path', type=str, required=True)
    argparser.add_argument('--model-type', type=str, required=True)
    # Other
    argparser.add_argument('--n-permutations', type=int, default=100000)
    # argparser.add_argument('--analyse', type=str, default='none',
    #                        choices=['none', 'vowels', 'consonants'])

    args = argparser.parse_args()
    args.keep_eos = args.model_type in ['norm', 'rev']
    # args.reverse = (args.model_type in constants.REVERSE_MODELS)
    # args.analyse = None if args.analyse == 'none' else args.analyse
    return args


def corrections(df, alpha):
    df.sort_values('p_value', inplace=True)
    df['count'] = range(1, df.shape[0] + 1)
    df['threshold'] = df['count'] * alpha / df.shape[0]
    df['significant'] = df['p_value'] <= df['threshold']
    last = df[df['significant']]['count'].max()
    df.loc[df.iloc[:last].index, 'significant'] = True

    del df['count']
    del df['threshold']

    return df


def main():
    args = get_args()
    alpha = 0.01

    print('\nRunning model: %s - %s' % (args.model_type, str(args.keep_eos)))

    fname = 'bin--%s_%s__%s--%d.tsv' % (args.dataset, args.model_type, str(args.keep_eos), args.n_permutations)
    df = pd.read_csv('results/p_values/%s' % fname, sep='\t')

    df = corrections(df, alpha)
    # print(df)
    # print("Average:", df['surp_avg'].mean())
    # # print(df['diff'].min())
    # # print(df['diff'].max())

    # # import ipdb; ipdb.set_trace()
    df_sign = df[df.significant]
    sign_neg = set(df_sign[df_sign['diff'] < 0].language.unique())
    sign_pos = set(df_sign[df_sign['diff'] > 0].language.unique())
    insign = set(df[~df.significant].language.unique())

    print('Significance Results:')
    print('\t# Significant Initial Languages: %d -- %s' % (len(sign_neg), str(sorted(sign_neg))))
    print('\t# Significant Final Languages: %d -- %s' % (len(sign_pos), str(sorted(sign_pos))))
    print('\t# Insignificant Languages: %d -- %s' % (df.shape[0] - len(sign_pos) - len(sign_neg), str(sorted(insign))))
    print('\t# Total Languages: %d' % df.shape[0])


if __name__ == '__main__':
    main()
