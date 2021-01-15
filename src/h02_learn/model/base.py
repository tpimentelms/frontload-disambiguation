import copy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from util import constants


class BaseLM(nn.Module, ABC):
    # pylint: disable=abstract-method,not-callable
    name = 'base'
    criterion_cls = None

    def __init__(self, alphabet):
        super().__init__()
        self.alphabet = alphabet

        self.best_state_dict = None
        self.alphabet_size = len(self.alphabet)
        self.pad_idx = alphabet.PAD_IDX
        self.eos_idx = alphabet.EOS_IDX

        self.criterion = self.criterion_cls(ignore_index=self.pad_idx) \
            .to(device=constants.device)
        self.criterion_full = self.criterion_cls(
            ignore_index=self.pad_idx, reduction='none') \
            .to(device=constants.device)

    def get_loss(self, y_hat, y):
        return self.criterion(
            y_hat.reshape(-1, y_hat.shape[-1]),
            y.reshape(-1))

    def get_loss_full(self, y_hat, y):
        return self.criterion_full(
            y_hat.reshape(-1, y_hat.shape[-1]),
            y.reshape(-1)) \
            .reshape_as(y)

    @abstractmethod
    def get_loss_no_eos(self, y_hat, y):
        pass

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path):
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    @abstractmethod
    def get_args(self):
        pass

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, path):
        fname = cls.get_name(path)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path):
        return '%s/model.tch' % (path)
