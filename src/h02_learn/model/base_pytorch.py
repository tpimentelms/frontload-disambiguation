import torch.nn as nn

from .base import BaseLM


class BasePytorchLM(BaseLM):
    # pylint: disable=abstract-method
    criterion_cls = nn.CrossEntropyLoss

    def __init__(self, alphabet, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__(alphabet)

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout

    def get_loss_no_eos(self, y_hat, y):
        y_hat[:, :, self.eos_idx] = -1e9
        return self.get_loss_full(y_hat, y)

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'alphabet': self.alphabet,
        }
