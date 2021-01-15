import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('./src/')
from h04_analysis.load_results import get_results
from h04_analysis.p_values_bin import get_dataframe
from util import argparser


def get_args():
    # Models
    argparser.add_argument('--checkpoints-path', type=str, required=True)

    return argparser.parse_args()


def get_df(lang, model_type, losses, y_values, lengths):
    df = get_dataframe(losses, y_values, lengths, model_type=model_type)
    df['lang'] = lang
    df['model_type'] = model_type

    return df


def get_df_full(losses, y_values, lengths):
    print('# Languages: %d\n' % len(losses))

    dfs = []
    for lang in losses.keys():
        losses_lang = losses[lang]
        for model_type in losses_lang.keys():
            dfs += [
                get_df(lang, model_type, losses[lang][model_type],
                       y_values[lang][model_type],
                       lengths[lang][model_type])
            ]

    dfs = pd.concat(dfs)
    return dfs


def get_data(models, eow_symbol, args):
    dfs = []
    for dataset in ['celex', 'wiki', 'northeuralex']:
        checkpoints_path = os.path.join(args.checkpoints_path, dataset)
        (losses, y_values, lengths) = get_results(checkpoints_path, keep_eos=True, models=models)

        df = get_df_full(losses, y_values, lengths)

        n_words = df.drop_duplicates(['idx', 'lang', 'model_type']).shape[0]
        n_eow = (df.char == eow_symbol).sum()
        assert (n_eow == n_words), 'Number of eow symbols should equal number of analysed words'

        df['is_eow'] = (df['char'] == eow_symbol)
        df = df.groupby(['is_eow', 'model_type', 'lang']).agg('mean')
        df = df.groupby(['is_eow', 'model_type']).agg('mean').reset_index()
        df['dataset'] = dataset

        dfs += [df]

    df = pd.concat(dfs)
    return df.groupby(['is_eow', 'model_type']).agg('mean').reset_index()


def print_eow(df, models, model_names):
    latex_str = '%s & %.2f & %.2f \\\\'
    for model in models:
        if model in ['position-nn', 'cloze']:
            eow_loss = 0
        else:
            eow_loss = df[(df.model_type == model) & (df.is_eow == True)].loss.item()
        neow_loss = df[(df.model_type == model) & (df.is_eow == False)].loss.item()

        print(latex_str % (model_names[model], eow_loss, neow_loss))


def main():
    args = get_args()
    eow_symbol = 2

    model_names = {
        'norm': 'Forward',
        'rev': 'Backward',
        'unigram': 'Unigram',
        'position-nn': 'Position',
        'cloze': 'Cloze',
    }
    models = ['norm', 'rev', 'unigram', 'position-nn', 'cloze']

    df = get_data(models, eow_symbol, args)
    print_eow(df, models, model_names)


if __name__ == '__main__':
    main()
