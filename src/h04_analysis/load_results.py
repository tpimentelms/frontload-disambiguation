import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.append('./src/')
from h02_learn.dataset import load_data, get_alphabet
from util import argparser
from util import util


def load_alphabet(data_path, lang):
    fname = os.path.join(data_path, lang, 'processed.pckl')
    data = load_data(fname)
    return get_alphabet(data)


def load_losses(lang, model_path, keep_eos=False):
    fname = 'losses.pckl'
    results_file = '%s/%s' % (model_path, fname)
    results = util.read_data(results_file)

    if not keep_eos:
        loss_value = 'losses_no_eos'
    else:
        loss_value = 'losses'

    loss = results['test'][loss_value].cpu().numpy() / math.log(2)
    y_values = results['test']['y_values'].cpu().numpy()
    if not keep_eos:
        mask = (y_values == 2)
        loss[mask] = 0
        y_values[mask] = 0

    lengths = (y_values != 0).sum(1)
    if keep_eos:
        lengths = lengths - 1

    return loss, y_values, lengths


def get_results(checkpoints_path, keep_eos=False, models=None):
    # pylint: disable=too-many-locals
    languages = util.get_dirs(checkpoints_path)

    if models is None:
        models = ['rev', 'norm', 'unigram', 'cloze', 'position-nn']

    losses = {}
    y_values = {}
    lengths = {}

    for lang_path in languages:

        lang = lang_path.split('/')[-1]

        losses[lang] = {}
        y_values[lang] = {}
        lengths[lang] = {}

        try:
            for model_type in models:
                model_path = os.path.join(lang_path, model_type)

                loss, y_value, length = load_losses(
                    lang, model_path, keep_eos=keep_eos)

                losses[lang][model_type] = loss
                y_values[lang][model_type] = y_value
                lengths[lang][model_type] = length

        except FileNotFoundError as err:
            print('Error loading lang %s results' % lang)

            del losses[lang]
            del y_values[lang]
            del lengths[lang]
            continue

    return (losses, y_values, lengths)
