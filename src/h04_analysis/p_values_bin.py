import os
import sys
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.append('./src/')
from h04_analysis.load_results import get_results
from util import argparser
from util import util


def get_args():
    # Models
    argparser.add_argument('--checkpoints-path', type=str, required=True)
    # Other
    argparser.add_argument('--analyse', type=str, default='none',
                           choices=['none', 'vowels', 'consonants'])
    argparser.add_argument('--n-permutations', type=int, default=100000)

    args = argparser.parse_args()
    args.analyse = None if args.analyse == 'none' else args.analyse
    return args


def get_pos(pos, length, model_type):
    if model_type == 'rev':
        return length - (pos + 1)
    return pos


def get_pos_bin(pos, length, model_type):
    if pos < (length - 1) / 2:
        which_bin = 0
    elif pos > (length - 1) / 2:
        which_bin = 1
    elif length == 1:
        which_bin = -5
    else:
        which_bin = -2

    if model_type == 'rev':
        return abs(which_bin - 1)
    return which_bin


def get_dataframe(losses, y_values, lengths, model_type, get_pos_func=get_pos):
    points = [
        {
            'loss': loss,
            'char': y_values[i, pos],
            'pos': get_pos_func(pos, lengths[i], model_type),
            'pos_bin': get_pos_bin(pos, lengths[i], model_type),
            'length': lengths[i],
            'word': tuple(y_values[i][:lengths[i]]),
            'idx': i,
        }
        for i, word in enumerate(losses)
        for pos, loss in enumerate(word)
        if y_values[i, pos] != 0
    ]

    return pd.DataFrame(points)


def get_bin_diffs(df, pos_column):
    groups = df.groupby(pos_column).agg('mean')
    means = groups['loss'].to_dict()
    diff = means[1] - means[0]

    return diff, means[0], means[1]


def get_p_value(df, n_permutations, step=1000):
    df = pd.pivot_table(df, values='loss', index=['idx'], columns=['pos_bin'], aggfunc=np.mean).reset_index()

    df = df.dropna()
    df['diff'] = df[1] - df[0]
    diff_avg = df['diff'].mean()

    diffs = df['diff'].values.copy().reshape(df.shape[0], 1)

    diff_permuts = []
    for _ in range(math.ceil(n_permutations / step)):
        signs = np.random.randint(0, 2, (df.shape[0], step)) * 2 - 1
        diff_permut = (diffs.repeat(step, 1) * signs).mean(0)
        diff_permuts += [diff_permut]
    diff_permuts = np.concatenate(diff_permuts)

    p_value = (np.abs(diff_permuts) >= abs(diff_avg)).mean()

    return p_value


def analyse_loss(lang, losses, y_values, lengths, n_permutations=100000, alpha=0.01, model_type='norm'):
    n_words, max_length = losses.shape
    df = get_dataframe(losses, y_values, lengths, model_type=model_type)
    df = df.groupby(['pos_bin', 'idx']).agg('mean').reset_index()

    df_word = df.groupby('idx').agg('mean')
    surp_avg = df_word.loss.mean()
    length_avg = df_word.length.mean()

    df = df[df.pos_bin.isin({0, 1})]
    diff, surp_initial, surp_final = get_bin_diffs(df, 'pos_bin')

    p_value = get_p_value(df, n_permutations=n_permutations)

    return diff, surp_initial, surp_final, surp_avg, length_avg, p_value


def analyse_language(lang, losses, y_values, lengths, n_permutations=100000, model_type='norm'):

    return analyse_loss(lang, losses[model_type], y_values[model_type], lengths[model_type], n_permutations=n_permutations, model_type=model_type)


def analyse_languages(losses, y_values, lengths, model_type='norm', n_permutations=1000):

    results = [['language', 'diff', 'surp_initial', 'surp_final', 'surp_avg', 'length_avg', 'p_value', 'n_permutations']]
    for lang in tqdm(losses.keys(), desc='Getting p_values'):
        diff, surp_initial, surp_final, surp_avg, length_avg, p_value = analyse_language(lang, losses[lang], y_values[lang], lengths[lang], n_permutations=n_permutations, model_type=model_type)
        results += [[lang, diff, surp_initial, surp_final, surp_avg, length_avg, p_value, n_permutations]]

    return results


def get_p_values(keep_eos, args, model):
    print('\nRunning model: %s - %s' % (model, str(keep_eos)))
    (losses, y_values, lengths) = get_results(args.checkpoints_path, keep_eos=keep_eos, models=[model])

    results = analyse_languages(losses, y_values, lengths, model_type=model, n_permutations=args.n_permutations)

    fname = '%s_%s__%s--%d.tsv' % (args.dataset, model, str(keep_eos), args.n_permutations)
    util.write_csv('results/p_values/bin--%s' % fname, results)


def main():
    args = get_args()
    models = ['position-nn', 'norm', 'cloze', 'rev', 'unigram']

    for model in models:
        for keep_eos in [False, True]:
            get_p_values(keep_eos, args, model)


if __name__ == '__main__':
    main()
