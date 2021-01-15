from tqdm import tqdm

from util import util
from .base import BaseDataProcesser


class Wikipedia(BaseDataProcesser):
    reverse_langs = ['ar']
    languages = [
        'af', 'ak', 'ar', 'bg', 'bn', 'chr', 'de', 'el', 'en', 'es',
        'et', 'eu', 'fa', 'fi', 'ga', 'gn', 'haw', 'he', 'hi', 'hu',
        'id', 'is', 'it', 'kn', 'lt', 'mr', 'no', 'nv', 'pl', 'pt',
        'ru', 'sn', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'tt', 'ur', 'zu'
    ]

    def __init__(self, fname, alphabet, max_words=None, max_len=50):
        super().__init__(fname, alphabet, max_words=max_words)

        self.max_len = max_len
        self.lang = self.fname.split('/')[-2]
        self.script = util.get_script(self.lang)

    def process_data(self):
        n_lines = util.get_n_lines(self.fname)

        word_info = {}
        with open(self.fname, 'r') as f:
            for line in tqdm(f, total=n_lines, desc='Processing wiki data'):
                self.process_line(line, word_info)

        print('Preview:')
        self.print_info(word_info)

        word_info = self.filter_data(word_info)
        print('Real:')
        self.print_info(word_info)

        return word_info

    def process_line(self, line, word_info):
        for word in line.strip().split(' '):
            if self.lang in self.reverse_langs:
                word = word[::-1]

            if len(word) > self.max_len:
                continue

            if not util.is_word(word, self.script):
                continue

            self.alphabet.add_word(word)

            if word in word_info:
                word_info[word]['count'] += 1
            else:
                word_info[word] = {
                    'count': 1,
                    'idx': self.alphabet.word2idx(word),
                    'word': word,
                    'tgt': word,
                }

    def filter_data(self, word_info):
        words_sorted = sorted(word_info.items(), key=lambda x: x[1]['count'], reverse=True)
        words_filtered = [
            (x, info) for x, info in words_sorted if x.strip() != '']

        if self.max_words:
            words_capped = words_filtered[:self.max_words]
        else:
            words_capped = words_filtered

        data = dict(words_capped)
        return data

    @classmethod
    def print_info(cls, words_info):
        n_tokens = cls.count_tokens(words_info)
        n_types = cls.count_types(words_info)

        print('# tokens:', n_tokens)
        print('# types:', n_types)

    @staticmethod
    def count_tokens(word_info):
        return sum([x['count'] for x in word_info.values()])

    @staticmethod
    def count_types(word_info):
        return len(word_info)

    @classmethod
    def get_languages(cls, *_):
        return cls.languages
