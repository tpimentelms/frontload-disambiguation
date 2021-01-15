import sys
import subprocess

sys.path.append('./src/')
from h01_data.dataset import get_languages
from util import argparser


def get_args():
    return argparser.parse_args()


def main():
    args = get_args()
    languages = get_languages(args.dataset, args.data_path)

    for i, lang in enumerate(languages):
        print()
        print('(%03d/%03d) Training on dataset: %s. Language: %s.' % \
              (i + 1, len(languages), args.dataset, lang))
        cmd = ['make',
               'LANGUAGE=%s' % (lang),
               'DATASET=%s' % (args.dataset)]
        print(cmd)
        subprocess.check_call(cmd)
        print()


if __name__ == '__main__':
    main()
