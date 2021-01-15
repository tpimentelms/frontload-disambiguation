
from .cloze import TransformerCloze
from .modules import TransformerLM


class PositionLM(TransformerCloze):
    # pylint: disable=too-few-public-methods

    def __init__(self, alphabet, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__(alphabet, embedding_size, hidden_size,
                         nlayers, dropout)

        self.sos_idx = self.alphabet.SOS_IDX

        self.transformer = TransformerLM(
            self.alphabet_size + 1, self.pad_idx,
            embedding_size, hidden_size,
            nlayers, dropout, tie_weights=False)

    def forward(self, x_orig):
        x = self.shift_input(x_orig)

        chars_mask = (x != self.eos_idx) & (x != self.pad_idx)
        x[chars_mask] = self.sos_idx

        return self.transformer(x)
