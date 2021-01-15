import torch
import torch.nn as nn

from .base import BaseLM


class UnigramLM(BaseLM):
    # pylint: disable=arguments-differ
    name = 'unigram'
    criterion_cls = nn.NLLLoss

    def __init__(self, alphabet):
        super().__init__(alphabet)

        self.probs = nn.Parameter(torch.Tensor(self.alphabet_size))
        self.log_probs = nn.Parameter(torch.Tensor(self.alphabet_size))
        self.count = torch.LongTensor(self.alphabet_size).zero_()

    def fit_batch(self, _, x):
        for char in x.unique():
            if char == self.pad_idx:
                continue

            self.count[char] += (x == char).sum()

        self.probs[:] = \
            (self.count.float() + 1) / (self.count.sum() + self.alphabet_size)
        self.log_probs[:] = torch.log(self.probs)

    def forward(self, x):
        batch_size, max_len = x.shape
        y_hat = self.log_probs \
            .reshape(1, 1, -1) \
            .repeat(batch_size, max_len, 1)

        return y_hat

    def get_loss_no_eos(self, y_hat, y):
        probs = y_hat.exp()
        probs[:, :, self.eos_idx] = 0
        probs = probs / probs.sum(-1, keepdim=True)
        y_hat = torch.log(probs)

        return self.get_loss_full(y_hat, y)

    def get_args(self):
        return {
            'alphabet': self.alphabet,
        }
