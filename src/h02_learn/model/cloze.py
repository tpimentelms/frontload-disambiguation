import torch

from util import constants
from .base_pytorch import BasePytorchLM
from .modules import TransformerLM


class TransformerCloze(BasePytorchLM):
    # pylint: disable=arguments-differ
    name = 'transformer-cloze'

    def __init__(self, alphabet, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__(alphabet, embedding_size, hidden_size,
                         nlayers, dropout)

        self.eos_idx = self.alphabet.EOS_IDX
        self.mask_idx = self.alphabet_size

        self.transformer = TransformerLM(
            self.alphabet_size + 1, self.pad_idx,
            embedding_size, hidden_size,
            nlayers, dropout)

    def forward(self, x_orig):
        batch_size, max_len = x_orig.shape
        x = self.shift_input(x_orig)
        return self.get_positional_logits(x, batch_size, max_len)

    def get_positional_logits(self, x, batch_size, max_len):
        logits = torch.zeros(batch_size, max_len, self.alphabet_size) \
            .to(device=constants.device)
        for position in range(max_len):
            logits_pos = self.get_positional_logit(x, position)

            logits[:, position] = logits_pos[:, position, :-1]

        return logits

    def get_positional_logit(self, x, position):
        x_pos = x.clone()
        x_pos[:, position] = self.mask_idx
        return self.transformer(x_pos)

    def shift_input(self, x_orig):
        assert self.eos_idx not in x_orig, \
            'EOS should never already be in the input'
        batch_size, max_len = x_orig.shape
        lengths = (x_orig != self.pad_idx).sum(-1)

        x = torch.zeros_like(x_orig)
        x[:, :max_len - 1] = x_orig[:, 1:]
        x[range(batch_size), lengths - 1] = self.eos_idx

        return x
