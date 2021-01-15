from .celex import Celex


class Syllex(Celex):
    # pylint: disable=too-few-public-methods

    @staticmethod
    def get_word(row):
        return row.phones, row.syllables.split('-')
