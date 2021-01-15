from .wikipedia import Wikipedia
from .northeuralex import Northeuralex
from .northeuragraph import NortheuraGraph
from .celex import Celex
from .syllex import Syllex


def get_dataset_cls(dataset_name):
    datasets = {
        'wiki': Wikipedia,
        'northeuralex': Northeuralex,
        'northeuragraph': NortheuraGraph,
        'celex': Celex,
        'syllex': Syllex,
    }
    return datasets[dataset_name]


def get_languages(dataset_name, data_path):
    dataset_cls = get_dataset_cls(dataset_name)
    return dataset_cls.get_languages(data_path)
