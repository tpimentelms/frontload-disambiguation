from .northeuralex import Northeuralex


class NortheuraGraph(Northeuralex):
    # pylint: disable=too-few-public-methods

    @staticmethod
    def get_word(row):
        return list(str(row.Word_Form))
