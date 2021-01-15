import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REVERSE_MODELS = ['rev', 'trie-rev']

scripts = {
    'af': ['Latin', 'Arabic'],
    'ak': ['Latin'],
    'bn': ['Bengali'],
    'chr': ['Cherokee', 'Latin'],
    'eu': ['Latin'],
    'fa': ['Arabic', 'Cyrillic', 'Hebrew'],
    'ga': ['Latin'],
    'gn': ['Latin'],
    'haw': ['Latin'],
    'hi': ['Devanagari'],
    'id': ['Latin'],
    'is': ['Latin'],
    'kn': ['Kannada', 'Brahmi'],
    'mr': ['Devanagari'],
    'no': ['Latin'],
    'nv': ['Latin'],
    'sn': ['Latin'],
    'sw': ['Latin', 'Arabic'],
    'ta': ['Tamil'],
    'te': ['Telugu'],
    'tl': ['Latin'],
    'tt': ['Latin', 'Cyrillic', 'Arabic'],
    'ur': ['Arabic'],
    'zu': ['Latin'],

    # 'ba': ['Cyrillic', 'Arabic'],
    # 'ceb': ['Latin'],
    # 'jv': ['Latin', 'Javanese'],
    # 'ne': ['Devanagari'],
    # 'yo': ['Latin', 'Arabic'],
    # 'cy': ['Latin'],
    # 'vo': ['Latin'],
    # 'ka': ['Georgian'],
    # 'az': None,
    # 'my': ['Myanmar'],
}

#  Default scripts: 'ar' 'bg' 'de', 'el' 'en', 'es' 'et'
#       'fi' 'he', 'hu' 'it' 'lt' 'pl' 'pt' 'ru' 'th' 'tr'
# 'Default Unused: lv' 'mk' 'nl' 'ro' 'sk' 'sl' 'sr' 'vi'
