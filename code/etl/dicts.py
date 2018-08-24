# -*- coding: utf-8 -*-
"""
A collection of dictionaries needed for data tidying on import of
the original dataset.

@author: Florian Hochstrasser
"""

from pandas.api.types import CategoricalDtype

# Several variables have NaN codes outside of what pandas interpretes as NaN.
# These have to be explicitly named on csv import through a dict.
NA_CODES = {
    'ODATEDW': '0',
    'TCODE': '000',
    'MDMAUD': 'XXXX',
    'DOB': '0',
    'AGE': '0',
    'CHILD03': ' ',
    'CHILD07': ' ',
    'CHILD12': ' ',
    'CHILD18': ' '}

# Boolean values are coded in several different ways in the original data.
# The following classes specify the mapping of the respective fields.


class BooleanRecodeConfig():
    """ Holds a dict of true / false value mappings and a list of fields
    in the data coded in that specific format."""

    def __init__(self, value_mapping, fields):
        self.value_mapping = value_mapping
        self.fields = fields

    def get_fields(self):
        """ Get all fields that are coded in the current mapping. """
        return self.fields

    def get_value_map(self):
        """ Get the recoding map."""
        return self.value_mapping


RECODE_X_UNDERSCORE = BooleanRecodeConfig({'true': 'X', 'false': '_'},
                                          ['NOEXCH',
                                           'RECINHSE',
                                           'RECP3',
                                           'RECPGVG',
                                           'RECSWEEP',
                                           ])

RECODE_BOOL_Y_N = BooleanRecodeConfig({'true': 'Y', 'false': 'N'},
                                      ['COLLECT1',
                                       'VETERANS',
                                       'BIBLE',
                                       'CATLG',
                                       'HOMEE',
                                       'PETS',
                                       'CDPLAY',
                                       'STEREO',
                                       'PCOWNERS',
                                       'PHOTO',
                                       'CRAFTS',
                                       'FISHER',
                                       'GARDENIN',
                                       'BOATS',
                                       'WALKER',
                                       'KIDSTUFF',
                                       'CARDS',
                                       'PLATES'
                                       ])

RECODE_BOOL_X_BLANK = BooleanRecodeConfig({'true': 'X', 'false': ' '},
                                          ['PEPSTRFL'])

BOOLEAN_RECODE = [RECODE_X_UNDERSCORE,
                  RECODE_X_UNDERSCORE,
                  RECODE_BOOL_X_BLANK]


# Donor title code. Pandas strips leading zeros on reading the data, so this is
# reflected in the index keys here. (1 instead of 001 and so on)
TCODE_CATEGORIES = {1: "MR.",
                    1001: "MESSRS.",
                    1002: "MR. & MRS.",
                    2: "MRS.",
                    2002: "MESDAMES",
                    3: "MISS",
                    3003: "MISSES",
                    4: "DR.",
                    4002: "DR. & MRS.",
                    4004: "DOCTORS",
                    5: "MADAME",
                    6: "SERGEANT",
                    9: "RABBI",
                    10: "PROFESSOR",
                    10002: "PROFESSOR & MRS.",
                    10010: "PROFESSORS",
                    11: "ADMIRAL",
                    11002: "ADMIRAL & MRS.",
                    12: "GENERAL",
                    12002: "GENERAL & MRS.",
                    13: "COLONEL",
                    13002: "COLONEL & MRS.",
                    14: "CAPTAIN",
                    14002: "CAPTAIN & MRS.",
                    15: "COMMANDER",
                    15002: "COMMANDER & MRS.",
                    16: "DEAN",
                    17: "JUDGE",
                    17002: "JUDGE & MRS.",
                    18: "MAJOR",
                    18002: "MAJOR & MRS.",
                    19: "SENATOR",
                    20: "GOVERNOR",
                    21002: "SERGEANT & MRS.",
                    22002: "COLNEL & MRS.",
                    24: "LIEUTENANT",
                    26: "MONSIGNOR",
                    27: "REVEREND",
                    28: "MS.",
                    28028: "MSS.",
                    29: "BISHOP",
                    31: "AMBASSADOR",
                    31002: "AMBASSADOR & MRS.",
                    33: "CANTOR",
                    36: "BROTHER",
                    37: "SIR",
                    38: "COMMODORE",
                    40: "FATHER",
                    42: "SISTER",
                    43: "PRESIDENT",
                    44: "MASTER",
                    46: "MOTHER",
                    47: "CHAPLAIN",
                    48: "CORPORAL",
                    50: "ELDER",
                    56: "MAYOR",
                    59002: "LIEUTENANT & MRS.",
                    62: "LORD",
                    63: "CARDINAL",
                    64: "FRIEND",
                    65: "FRIENDS",
                    68: "ARCHDEACON",
                    69: "CANON",
                    70: "BISHOP",
                    72002: "REVEREND & MRS.",
                    73: "PASTOR",
                    75: "ARCHBISHOP",
                    85: "SPECIALIST",
                    87: "PRIVATE",
                    89: "SEAMAN",
                    90: "AIRMAN",
                    91: "JUSTICE",
                    92: "MR. JUSTICE",
                    100: "M.",
                    103: "MLLE.",
                    104: "CHANCELLOR",
                    106: "REPRESENTATIVE",
                    107: "SECRETARY",
                    108: "LT. GOVERNOR",
                    109: "LIC.",
                    111: "SA.",
                    114: "DA.",
                    116: "SR.",
                    117: "SRA.",
                    118: "SRTA.",
                    120: "YOUR MAJESTY",
                    122: "HIS HIGHNESS",
                    123: "HER HIGHNESS",
                    124: "COUNT",
                    125: "LADY",
                    126: "PRINCE",
                    127: "PRINCESS",
                    128: "CHIEF",
                    129: "BARON",
                    130: "SHEIK",
                    131: "PRINCE AND PRINCESS",
                    132: "YOUR IMPERIAL MAJEST",
                    135: "M. ET MME.",
                    210: "PROF."}


# Categorical variables have to be handled on import. List every one here.
# By passing a CategoricalDtype, the categories can be setup correctly.
# Any level not present will be coded as NaN.
# categoricals are created unordered by default., however when implicitly
# constructing one, the argument ordered=False has to be set for unordered.
DTYPE_CATEGORICAL = {
    'TCODE': CategoricalDtype(categories=TCODE_CATEGORIES,
                              ordered=False),
    'STATE': 'category',
    'MAILCODE': 'category',
    'PVASTATE': 'category',
    'CLUSTER': 'category',
    'AGEFLAG': 'category',
    'HOMEOWNR': 'category',
    'CHILD03': 'category',
    'CHILD07': 'category',
    'CHILD12': 'category',
    'CHILD18': 'category',
    'GENDER': 'category',
    'WEALTH1': 'category',
    'DATASRCE': 'category',
    'SOLP3': 'category',    # THESE NEED A RECATEGORIZATION! NaN MEANS SOMETHING TOO!
    'SOLIH': 'category',    # THESE NEED A RECATEGORIZATION! NaN MEANS SOMETHING TOO!
    'WEALTH2': 'category',
    'LIFESRC': 'category',

}
