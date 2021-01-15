import os
import sys
import pandas as pd

sys.path.append('./src/')
from util import argparser


def get_args():
    # Other
    argparser.add_argument('--n-permutations', type=int, default=100000)
    return argparser.parse_args()


def main():
    args = get_args()

    model_names = {
        'norm': 'Forward',
        'rev': 'Backward',
        'unigram': 'Unigram',
        'position-nn': 'Position',
        'cloze': 'Cloze',
    }
    datasets = ['celex', 'northeuralex', 'wiki']
    models = ['norm', 'rev', 'unigram', 'position-nn', 'cloze']

    latex_str = '%s & %s & %s & %s & %s & %s & %s \\\\'

    for model in models:
        results = [model_names[model]]
        for keep_eos in [True, False]:
            if model in ['unigram', 'position-nn', 'cloze'] and keep_eos == True:
                results += ['-', '-', '-']
                continue

            dfs = []
            for dataset in datasets:
                fname = 'bin--%s_%s__%s--%d.tsv' % (dataset, model, str(keep_eos), args.n_permutations)
                df = pd.read_csv('results/p_values/%s' % fname, sep='\t')

                df['diff'] = - df['diff']
                df['dataset'] = dataset

                dfs += [df]

            df = pd.concat(dfs)

            results += [
                '%.2f' % df['surp_initial'].mean(), '%.2f' % df['surp_final'].mean(),
                '%.1f \\%%' % (
                    100 * df['diff'].mean() /
                    max(df['surp_initial'].mean(), df['surp_final'].mean())
                )]

        print(latex_str % tuple(results))


if __name__ == '__main__':
    main()
