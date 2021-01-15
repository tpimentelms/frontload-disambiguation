import torch

from torch.utils.data import Dataset


class TypeDataset(Dataset):
    # pylint: disable=no-member,too-many-instance-attributes

    def __init__(self, data, folds, reverse=False):
        self.data = data
        self.folds = folds
        self.reverse = reverse
        self.process_train(data, reverse=reverse)
        self._train = True

    def process_train(self, data, reverse=False):
        folds_data = data[0]
        self.alphabet = data[1]

        folds_words = [instance['tgt'] for fold in self.folds for instance in folds_data[fold]]
        if not reverse:
            self.words = folds_words
        else:
            self.words = [list(word)[::-1] for word in folds_words]

        self.word_train = [torch.LongTensor(self.get_word_idx(word)) for word in self.words]
        self.n_instances = len(self.word_train)

    def get_word_idx(self, word):
        return [self.alphabet.char2idx('SOS')] + \
            self.alphabet.word2idx(word) + \
            [self.alphabet.char2idx('EOS')]

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return (self.word_train[index],)
