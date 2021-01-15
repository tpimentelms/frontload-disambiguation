# import re
# from tqdm import tqdm
import pandas as pd

from h01_data.alphabet import Alphabet
from .base import BaseDataProcesser


class Celex(BaseDataProcesser):
    languages = [
        'eng', 'deu', 'nld'
    ]

    def process_data(self):
        df = pd.read_csv(self.fname, delimiter='\t')
        self.pos_classes = Alphabet()

        word_info = {}

        for _, row in df.iterrows():
            self.process_row(row, word_info)

        return word_info

    def process_row(self, row, word_info):
        tgt_word, tgt_splits = self.get_word(row)
        pos_tag = row.pos

        self.alphabet.add_word(tgt_splits)
        self.pos_classes.add_char(pos_tag)

        word_info[tgt_word] = {
            'count': row.freq,
            'idx': self.alphabet.word2idx(tgt_splits),
            'word': tgt_word,
            'tgt': tgt_splits,
            'grapheme': row.word,
            'pos': pos_tag,
            'pos_idx': self.pos_classes.char2idx(pos_tag),
        }

    @staticmethod
    def get_word(row):
        return row.phones, list(row.phones)

    @classmethod
    def get_languages(cls, *_):
        return cls.languages
