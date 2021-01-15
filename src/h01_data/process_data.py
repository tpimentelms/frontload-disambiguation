import sys
import logging

sys.path.append('./src/')
from h01_data.alphabet import Alphabet
from h01_data.dataset import get_dataset_cls
from util import argparser


def get_args():
    argparser.add_argument(
        "--src-file", type=str,
        help="The file from which to read data")
    argparser.add_argument(
        "--n-folds", type=int, default=10,
        help="Number of folds to split data")
    argparser.add_argument(
        "--max-words", type=int, default=10000,
        help="Number of types to use")

    return argparser.parse_args()


def get_dataset(dataset_name, src_fname, alphabet, max_words):
    dataset_cls = get_dataset_cls(dataset_name)
    return dataset_cls(src_fname, alphabet, max_words)


def process(src_fname, dataset, tgt_fname, n_folds, max_words):
    alphabet = Alphabet()

    dataset = get_dataset(dataset, src_fname, alphabet, max_words)
    words_info = dataset.process_data()
    splits = dataset.get_fold_splits(words_info, n_folds)

    dataset.write_data(tgt_fname, splits)

    print('# unique chars:', len(alphabet))


def main():
    args = get_args()
    logging.info(args)

    process(args.src_file, args.dataset, args.data_path, args.n_folds, args.max_words)


if __name__ == '__main__':
    main()
