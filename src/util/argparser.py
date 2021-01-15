import argparse
from util import util

parser = argparse.ArgumentParser(description='LanguageModel')
# Data Preprocess
parser.add_argument('--data-path', type=str)

# Data
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--reverse', action='store_true', default=False)
parser.add_argument('--data-file', type=str)
# Model
parser.add_argument('--model', default='lstm', choices=['lstm'],
                    help='Model used. (default: lstm)')
# Others
parser.add_argument('--seed', type=int, default=7,
                    help='Seed for random algorithms repeatability (default: 7)')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)

    util.config(args.seed)
    return args
