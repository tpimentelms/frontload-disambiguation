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
from util import argparser


aspect_single = {
    'font_scale': 2.5,
    'labels': False,
    'name_suffix': 'small__shared',
    'ratio': 2.5,
    'width': 6.5 * 1.625,
}

aspect = {
    'font_scale': 2.5,
    'labels': False,
    'name_suffix': 'small__shared',
    'ratio': 1.625,
    'width': 6.5,
}
aspect['height'] = aspect['width'] / aspect['ratio']

cmap = sns.color_palette("PRGn", 14)
sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
mpl.rc('figure', figsize=(aspect['width'], aspect['height'] * 1.3))
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})


def get_args():
    # Other
    argparser.add_argument('--n-permutations', type=int, default=100000)
    argparser.add_argument('--results-path', type=str, default='results/')

    return argparser.parse_args()


def plot_languages_new(df, results_path, model):
    hue = 'Name'
    y_value = 'Surprisal'
    x_value = 'Position'

    x_label = 'Initial Surprisal (bits)'
    y_label = 'Final Surprisal (bits)'

    df[x_label] = df['surp_initial']
    df[y_label] = df['surp_final']
    markers = {'celex': 's', 'northeuralex': 'X', 'wikipedia': '.'}

    min_x = min(df[x_label].min(), df[y_label].min())
    max_x = max(df[x_label].max(), df[y_label].max())
    fig = sns.scatterplot(x=x_label, y=y_label, data=df, hue='dataset', style='dataset', s=200, alpha=.8)
    plt.plot([min_x, max_x], [min_x, max_x], '--', color='C7', alpha=.5, linewidth=3)

    handles, labels = fig.get_legend_handles_labels()
    fig.legend(handles=handles[1:], labels=labels[1:], handletextpad=0, borderpad=0.2, borderaxespad=0.1, labelspacing=0.1)

    fname = 'surprisal--diff_%s.pdf' % model
    fname = os.path.join(results_path, fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def get_data_an_plot(args, model, keep_eos):
    datasets = ['celex', 'northeuralex', 'wiki']
    dataset_name = {'wiki': 'wikipedia'}

    dfs = []
    for dataset in datasets:
        fname = 'bin--%s_%s__%s--%d.tsv' % (dataset, model, str(keep_eos), args.n_permutations)
        df = pd.read_csv('results/p_values/%s' % fname, sep='\t')
        df['dataset'] = dataset_name[dataset] if dataset in dataset_name else dataset
        dfs += [df]

    df = pd.concat(dfs)
    df.sort_values('dataset', ascending=False, inplace=True)

    df['diff'] = - df['diff']
    df = plot_languages_new(df, args.results_path, model)


def main():
    args = get_args()
    model = 'cloze'

    keep_eos = False
    for model in ['cloze', 'unigram', 'position-nn']:
        get_data_an_plot(args, model, keep_eos)

    keep_eos = True
    for model in ['norm', 'rev']:
        get_data_an_plot(args, model, keep_eos)


if __name__ == '__main__':
    main()
