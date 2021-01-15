from .lstm import LstmLM
from .unigram import UnigramLM
from .cloze import TransformerCloze
from .position_nn import PositionLM


def get_model_cls(model_type):
    if model_type == 'unigram':
        model_cls = UnigramLM
    elif model_type in ['norm', 'rev']:
        model_cls = LstmLM
    elif model_type == 'cloze':
        model_cls = TransformerCloze
    elif model_type == 'position-nn':
        model_cls = PositionLM
    else:
        raise ValueError('Not implemented: %s' % model_type)

    return model_cls
