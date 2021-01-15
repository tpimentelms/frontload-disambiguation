
class Alphabet:
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self):
        self._chars2idx = {
            'PAD': 0,
            'SOS': 1,
            'EOS': 2
        }
        self._chars_count = {
            'PAD': 0,
            'SOS': 0,
            'EOS': 0,
        }
        self._idx2chars = {idx: char for char, idx in self._chars2idx.items()}
        self._updated = True


    def add_char(self, char):
        if char not in self._chars2idx:
            self._chars2idx[char] = len(self._chars2idx)
            self._chars_count[char] = 1
            self._updated = False
        else:
            self._chars_count[char] += 1

    def add_word(self, word):
        for char in word:
            self.add_char(char)

    def word2idx(self, word):
        return [self._chars2idx[char] for char in word]

    def char2idx(self, char):
        return self._chars2idx[char]

    def idx2word(self, idx_word):
        if not self._updated:
            self._idx2chars = {idx: char for char, idx in self._chars2idx.items()}
            self._updated = True
        return [self._idx2chars[idx] for idx in idx_word]

    def idx2char(self, idx_char):
        if not self._updated:
            self._idx2chars = {idx: char for char, idx in self._chars2idx.items()}
            self._updated = True
        return self._idx2chars[idx_char]

    def __len__(self):
        return len(self._chars2idx)

    def items(self):
        for chars, idx in self._chars2idx.items():
            yield chars, idx
