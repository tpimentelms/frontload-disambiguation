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
from util import argparser
from util import util

aspect = {
    # 'size': 6.5,
    'font_scale': 2.5,
    # 'ratio': 1.625,
    # 'ratio': 1.625 / 1.5,
    'ratio': 1.625 / 1.3,
    'width': 6.5 * 1.625,
}
aspect['height'] = aspect['width'] / aspect['ratio']

cmap = sns.color_palette("PRGn", 14)
sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
mpl.rc('figure', figsize=(aspect['width'], aspect['height']))
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})


def get_args():
    # Models
    argparser.add_argument('--checkpoints-path', type=str, required=True)
    # Results
    argparser.add_argument('--results-path', type=str, default='results/')

    return argparser.parse_args()


def get_dataframe(losses, y_values, lengths, model_type='norm'):
    max_length = max(lengths)
    points = [
        {
            'loss': loss,
            'pos_split': pos if model_type not in ['rev', 'trie-rev'] else -pos,
            'idx': i,
        }
        for i, word in enumerate(losses)
        for pos, loss in enumerate(word)
        if y_values[i, pos] != 0
    ]

    return pd.DataFrame(points)


def get_dataframe_dataset(losses, y_values, lengths, model_types):
    dfs = []

    for lang in losses.keys():
        for model_type in model_types:
            df = get_dataframe(losses[lang][model_type], y_values[lang][model_type], lengths[lang][model_type], model_type)
            df['lang'] = lang
            df['model_type'] = model_type

            dfs += [df]

    return pd.concat(dfs)


def get_dataframe_full(models, args, keep_eos=True):
    dfs = []
    dataset_name = {'wiki': 'wikipedia'}

    for dataset in ['celex', 'wiki', 'northeuralex']:
        data_path = os.path.join(args.data_path, dataset)
        checkpoints_path = os.path.join(args.checkpoints_path, dataset)

        (losses, y_values, lengths) = get_results(checkpoints_path, keep_eos=keep_eos, models=models)
        df = get_dataframe_dataset(losses, y_values, lengths, model_types=models)
        df['dataset'] = dataset_name[dataset] if dataset in dataset_name else dataset

        dfs += [df]

    return pd.concat(dfs)


def format_dataframe(df):
    label = {
        'norm': 'standard',
        'rev': 'reversed',
    }

    y_value = 'Surprisal'

    df[y_value] = df['loss']
    df['model'] = df['model_type'].apply(lambda x: label[x])

    return df


def plot_languages_split(df, results_path, model_type, ax):
    hue = 'Dataset'

    if model_type in ['rev']:
        y_value = 'Backward'
        x_value = 'Position from End'
    else:
        y_value = 'Forward'
        x_value = 'Position from Start'

    df = df[df['model_type'] == model_type].copy()
    df[hue] = df['dataset']
    df[y_value] = df['Surprisal']
    df[x_value] = df['pos_split']

    lengths = np.sort(df[x_value].values)
    if model_type in ['rev']:
        x_len = lengths[int(0.1 * len(lengths)) - 1]
        xlim = [x_len - .25, 0.25]
    else:
        x_len = lengths[int(0.9 * len(lengths)) + 1]
        xlim = [-0.25, x_len + .25]

    lines, legends = [], []
    for mode in df[hue].unique():
        df_temp = df[df[hue] == mode]
        sns.regplot(df_temp[x_value], df_temp[y_value], ax=ax, x_bins=10, label=mode)
        legends += [mode]

    ax.set_xlim(xlim)
    ax.legend(handletextpad=0, borderpad=0.1, borderaxespad=0.2, labelspacing=0.1)

    return lines, legends


def plot_langs(df, models, results_path):
    name = {
        'trie': 'trie',
        'norm': 'lstm',
    }
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharey=True)
    plt.subplots_adjust(hspace=.7)

    plot_languages_split(df, results_path, models[0], ax=ax1)
    lines, legends = plot_languages_split(df, results_path, models[1], ax=ax2)

    plt.ylim([1.3, 5])

    fname = 'surprisal--forward_backward.pdf'
    fname = os.path.join(results_path, fname)
    fig.savefig(fname, bbox_inches='tight')


def main():
    args = get_args()

    models = ['norm', 'rev']
    df = get_dataframe_full(models, args, keep_eos=True)

    df = format_dataframe(df)
    plot_langs(df, models, args.results_path)


if __name__ == '__main__':
    main()
