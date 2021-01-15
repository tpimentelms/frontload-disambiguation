import os
import pandas as pd

from h01_data.alphabet import Alphabet
from util import util
from .base import BaseDataProcesser


class Northeuralex(BaseDataProcesser):
    def process_data(self):
        df = self.read_data(self.fname)
        self.languages = df.Language_ID.unique()
        self.alphabets = {lang: Alphabet() for lang in self.languages}
        self.pos_classes = {lang: Alphabet() for lang in self.languages}

        word_info = {}
        for _, row in df.iterrows():
            self.process_row(row, word_info)

        return word_info

    @staticmethod
    def read_data(fname):
        df = pd.read_csv(fname, delimiter='\t')
        df['id'] = range(df.shape[0])

        return df

    def process_row(self, row, word_info):
        word = self.get_word(row)
        lang = row.Language_ID

        self.alphabet.add_word(word)
        self.alphabets[lang].add_word(word)
        idx = self.alphabets[lang].word2idx(word)

        pos_tag = row.Concept_ID.split(':')[-1]
        self.pos_classes[lang].add_char(pos_tag)
        pos_idx = self.pos_classes[lang].char2idx(pos_tag)

        if row.Concept_ID not in word_info:
            word_info[row.Concept_ID] = []

        word_info[row.Concept_ID] += [{
            'count': 1,
            'idx': idx,
            'concept': row.Concept_ID,
            'tgt': word,
            'word': ''.join(word),
            'id': row.id,
            'grapheme': row.Word_Form,
            'lang': lang,
            'pos': pos_tag,
            'pos_idx': pos_idx,
        }]

    @staticmethod
    def get_word(row):
        return str(row.IPA).split(' ')

    def write_data(self, tgt_path, splits):
        data = {lang: [[] for _ in splits] for lang in self.languages}

        for i, split in enumerate(splits):
            for _, datum in split.items():
                for instance in datum:
                    lang = instance['lang']
                    data[lang][i] += [
                        instance
                    ]

        for lang, datum in data.items():
            lang_path = os.path.join(tgt_path, lang)
            util.mkdir(lang_path)

            lang_fname = os.path.join(lang_path, 'processed.pckl')
            util.write_data(lang_fname, (data[lang], self.alphabets[lang], self.pos_classes[lang]))

        tgt_fname = os.path.join(tgt_path, 'processed.pckl')
        util.write_data(tgt_fname, (data, self.alphabet))

    @classmethod
    def get_languages(cls, data_path):
        fname = os.path.join(data_path, 'northeuralex-0.9-forms.tsv')
        df = cls.read_data(fname)
        return df.Language_ID.unique()
