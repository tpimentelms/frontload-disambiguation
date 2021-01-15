import os
import pathlib
import io
import csv
import random
import pickle
from tqdm import tqdm
import homoglyphs as hg
import numpy as np
import torch

from util import constants


def config(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def write_csv(filename, results):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(results)


def write_data(filename, embeddings):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def read_data(filename):
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def read_data_if_exists(filename):
    try:
        return read_data(filename)
    except FileNotFoundError:
        return {}


def remove_if_exists(fname):
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass


def get_filenames(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isfile(os.path.join(filepath, f))]
    return sorted(filenames)


def get_dirs(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isdir(os.path.join(filepath, f))]
    return sorted(filenames)


def get_n_lines(fname):
    with open(fname, 'r') as file:
        count = 0
        for _ in tqdm(file, desc='Getting file length'):
            count += 1
    return count


def get_script(language):
    try:
        alphabet = hg.Languages.get_alphabet([language])
    except ValueError:
        if constants.scripts[language] is not None:
            alphabet = hg.Categories.get_alphabet(
                [x.upper() for x in constants.scripts[language]]
            )
        else:
            alphabet = None

    return alphabet


def is_word(token, alphabet):
    return all([x in alphabet for x in token])


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
