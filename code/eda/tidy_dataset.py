# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import numpy as np
import pandas as pd
#from pandas.api.types import CategoricalDtype
from config import App


# Boolean values are coded in several different ways in the original data.
# The following class specifys the mapping of the respective fields.

class BooleanRecodeSpec:
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


class TidyDataset:
    """
    Represents a tidy dataset for either training or test data of the
    kdd cup 1998. This class is recommended to load ready-to-work-with data.
    Expects input data as distributed on UCI's machine learning repository
    (either 'cup98LRN.txt' or 'cup98VAL.txt')."
    """

    ###########################################################################
    # Dicts and data structures to recode / reformat various variables
    ###########################################################################

    # Several variables have NaN codes outside of what pandas interpretes as NaN.
    # These have to be explicitly named on csv import through a dict.
    na_codes = {
        'ODATEDW': '0',
        'TCODE': '000',
        'MDMAUD': 'XXXX',
        'DOB': '0',
        'AGE': '0',
        'CHILD03': ' ',
        'CHILD07': ' ',
        'CHILD12': ' ',
        'CHILD18': ' '}

    recode_x_underscore = BooleanRecodeSpec({'true': 'X', 'false': '_'},
                                            ['NOEXCH',
                                             'RECINHSE',
                                             'RECP3',
                                             'RECPGVG',
                                             'RECSWEEP',
                                             ])

    recode_y_n = BooleanRecodeSpec({'true': 'Y', 'false': 'N'},
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

    recode_x_blank = BooleanRecodeSpec({'true': 'X', 'false': ' '},
                                       ['PEPSTRFL'])

    boolean_recode = [recode_x_underscore,
                      recode_y_n,
                      recode_x_blank]

    # Donor title code. Pandas strips leading zeros on reading the data,
    # so this is reflected in the index keys here. (1 instead of 001 and so on)
    tcode_categories = {
        1: "MR.",
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
    # TODO: SOLP3 and SOLIH have NaN as a level too!
    dtype_categorical = {
        'TCODE': 'category',
        'STATE': 'category',
        'MAILCODE': 'category',
        'PVASTATE': 'category',
        'MDMAUD': 'str',
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
        'SOLP3': 'category',
        'SOLIH': 'category',
        'WEALTH2': 'category',
        'LIFESRC': 'category'
    }

    index_field = "CONTROLN"
    target_vars = ["TARGET_B", "TARGET_D"]

    data_path = App.config("data_dir")

    def __init__(self, csv_file=None, pull_stored=True):
        """
        Initializes a new object. No data is pulled rightaway. Magically
        initializes certain fields based on the occurrence of either of the
        strings ['LRN','VAL'] in the filename. So please do not rename the
        files (as distributed on the UCI repository) arbitrarily.

        Parameters:
        -----------
        csv_file: The name of either the training or test file (cup98[LRN,VAL].txt)
        pull_stored: Whether to attempt loading from hdf store before reading
            in csv data (default True).
        """
        self.hdf_data_file = App.config("hdf_store")
        self.pull_stored = pull_stored
        self.raw_data_file = csv_file
        self.hdf_store = App.config("data_dir")+App.config("hdf_store")
        self.raw_data = None
        self.processed_data = None

        if not csv_file is None and csv_file in [App.config("train_file_name"),
                                                 App.config("test_file_name")]:
            if "lrn" in csv_file.lower():
                self.dataset_type = App.config("train_name")
            elif "val" in csv_file.lower():
                self.dataset_type = App.config("test_name")
        else:
            raise NameError("Set csv_file to either training- or test-file.")

    def _four_digit_date_parser(self, date):
        """
        Formats YYMM dates as YYYY-MM-DD where DD is the first d.o.m always.
        """
        if len(date) == 4:
            parsed_date = pd.to_datetime(date, format="%y%m")
        else:
            parsed_date = np.nan

        return parsed_date

    def _recode_booleans(self, data, recode_specs):
        """
        Recodes boolean columns. Specify the codes used in data and the
        affected columns through BooleanRecodeSpec objects.
        Expects a pandas data frame!
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Needs a pandas dataframe.")

        if not all(isinstance(r, BooleanRecodeSpec) for r in recode_specs):
            raise TypeError("Expects a list of BooleanRecodeSpec objects.")

        def do_recode(recode_spec):

            true_char = recode_spec.value_mapping.get('t')
            false_char = recode_spec.value_mapping.get('f')
            try:
                for field in recode_spec.fields:
                    name = str(field)
                    data.loc[data[name] == false_char, name] = False
                    data.loc[data[name] == true_char, name] = True
                    data.loc[(data[name] != true_char) & (
                        data[name] != false_char), name] = np.nan
            except Exception as exc:
                print("Failed to recode boolean:")
                print(exc)

        for spec in recode_specs:
            do_recode(spec)

    def _split_promotion_history(self):
        # TODO: Implement, along with other vars that need splitting
        """
        The promotion history data is aggregated
        """
        pass

    def _read_csv_data(self):
        """ Read in csv data. """
        try:
            self.raw_data = pd.read_csv(self.get_raw_datafile_path(),
                                        index_col=self.index_field,
                                        parse_dates=[0, 7],
                                        date_parser=self._four_digit_date_parser,
                                        na_values=self.na_codes,
                                        dtype=self.dtype_categorical,
                                        low_memory=False,  # needed for mixed type columns
                                        memory_map=True  # load file in memory
                                        )
        except Exception as exc:
            print(exc)
            raise

    def _process_raw(self):
        """
        Processes the raw csv import.
            - Recodes booleans
            - Splits multi-value columns
            - Renames categorical columns
        """
        if self.raw_data is None:
            self._read_csv_data()
        try:
            self.processed_data = self.raw_data
            self._recode_booleans(self.processed_data, self.boolean_recode)
        except Exception as exc:
            self.processed_data = None
            print(exc)

    def _load_hdf(self):
        """ Loads tidy data from hdf store """

        # If data should not be pulled, throw an exception to make sure
        # data is loaded from csv
        if not self.pull_stored:
            raise ValueError("HDF loading prohibited by options set.")

        try:
            self.processed_data = pd.read_hdf(self.get_hdf_datafile_path(),
                                              key=self.dataset_type,
                                              mode='r')
        except (OSError, IOError):
            raise

    def _save_hdf(self):
        """ Save tidy data to hdf store """
        try:
            self.processed_data.to_hdf(self.get_hdf_datafile_path(),
                                       key=self.dataset_type,
                                       format='table')
        except Exception as exc:
            print(exc)

    def _load_data(self):
        """
        Makes data available. First, checks if processed_data is alread present.
        If not, attempts to load from hdf, failing that, reads in csv and
        processes the data
        """
        if self.processed_data is None:
            try:
                self._load_hdf()
            except:
                if self.raw_data is None:
                    try:
                        self._read_csv_data()
                    except Exception as exc:
                        print(exc)
                self._process_raw()
                self._save_hdf()

    def get_raw_datafile_path(self):
        """ Return relative path to csv data file"""
        if self.raw_data_file is None:
            raise NameError("No data file specified yet.")
        return self.data_path+self.raw_data_file

    def get_hdf_datafile_path(self):
        """ Return relative path to hdf data file"""
        return self.hdf_store

    def get_target_variables(self, names_only=True):
        """
        Returns the target variables.

        TARGET_B is the response to the mailing: 1 - has donated

        TARGET_D is the dollar amount donated

        Parameters
        ----------
        names_only: If set to false, returns the complete series (default True)

        Returns
        -------
        pandas.Dataframe object, possibly empty (if names_only=True)
            with columns TARGET_B and TARGET_D
        """

        if names_only:
            return pd.DataFrame(columns=App.config("dependent_vars"))

        if self.processed_data is None:
            self._load_data()
        return self.processed_data.loc[:, App.config("dependent_vars")]

    def get_data(self):
        """
        Gets processed data ready for further analysis. Attempts to load
        hdf, if that's not available, reads csv and processes data first.

        Returns
        -------
        processed_data: pandas.DataFrame object with tidy data
        """
        if self.processed_data is None:
            self._load_data()

        return self.processed_data
