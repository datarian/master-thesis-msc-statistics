# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import numpy as np
import pandas as pd
import traceback
import logging
from config import App

# Set up the logger
logging.basicConfig(filename=__name__+'.log',level=logging.WARNING)
logger = logging.getLogger(__name__)


class BooleanRecodeSpec:
    """ Holds a dict of true / false value mappings and a list of fields
    in the data coded in that specific format."""

    def __init__(self, value_mapping, fields):
        self.value_mapping = value_mapping
        self.fields = fields
        self.tds_ref = None # Reference to a TidyDataset object

        @property
        def tds_ref(self):
            return(self.tds_ref)

        @tds_ref.setter
        def tds_ref(self, ref):
            self.tds_ref = ref

    def get_fields(self):
        """ Get all fields that are coded in the current mapping. """
        return self.fields

    def get_value_map(self):
        """ Get the recoding map."""
        return self.value_mapping

    def do_recode(self):

            if not self.tds_ref:
                raise ValueError("Missing object reference.")
            true_char = self.value_mapping.get('true')
            false_char = self.value_mapping.get('false')
            try:
                for field in self.fields:
                    self.tds_ref.processed_data[field] = self.tds_ref.processed_data[field].map({true_char: True, false_char: False})
                    self.tds_ref.processed_data[field].fillna(False,inplace=True) #everything that's not coded as True or False already is set to False
                    self.tds_ref.processed_data[field].astype('bool',copy=False)
            except Exception as exc:
                logger.exception(exc)


class SymbolicFieldSpreader:
    """
    Holds information on a symbolic field and the functions used
    to create dummy variables from it.

    Directly modifies the dataset passed to the class!

    Params:
    -------
    tidy_dataset: A reference to the TidyDataset class
    field The field name to split
    field_names The identifiers for each byte, from left to right
    """

    def __init__(self, field, field_names):
        self.tds_ref = None
        self.field = field
        self.newfields = field_names
        self.original_field = None
        self.temp_df = pd.DataFrame()

    def set_tidy_dataset_ref(self, tidy_dataset):
        self.tds_ref = tidy_dataset
        self.original_field = pd.DataFrame(self.tds_ref.processed_data.loc[:, self.field]) # The original field to be replaced


    def _fill_temp_with_bytes(self):
        """ Fills the byte dataset for each record"""

        # Make sure the reference to the tidy dataset instance is around:
        if not self.tds_ref:
            raise ValueError("No reference to dataset found!")

        sigbytes = len(self.newfields) # determines how many bytes to extract
        # Dict to hold the split bytes
        spread_field = {}

        # Iterate over all rows, fill into dict
        for row in self.original_field.itertuples(name=None):
            # row[0] is the index, row[1] the content of the cell
            if not row[1] is np.nan:
                if len(row[1]) == sigbytes:
                    spread_field[row[0]] = list(row[1])
                else:
                    # The field is invalid, set all to empty
                    spread_field[row[0]] = [np.nan]*sigbytes
            else:
                # handle missing values -> set all empty
                spread_field[row[0]] = [np.nan]*sigbytes

        # Create the dataframe, orient=index means we interprete the dict's contents as rows (defaults to columns)
        self.temp_df = pd.DataFrame.from_dict(data=spread_field, orient="index")
        self.temp_df.columns = ["_".join([self.field,f]) for f in self.newfields]
        self.temp_df.index.name = "CONTROLN"
        self.temp_df = self.temp_df.astype("category") # make sure all fields are categorical


    def spread(self):
        """Spreads a symbolic field bytewise into categoricals, then drops the initial symbolic field."""
        #dummies = self._create_dummies()
        self._fill_temp_with_bytes()
        self.tds_ref.processed_data = self.tds_ref.processed_data.merge(self.temp_df,on="CONTROLN",copy=False)
        self.tds_ref.processed_data.drop(self.field, axis=1, inplace=True)


class TidyDataset:
    """
    Represents a tidy dataset for either learning or validation data of the
    kdd cup 1998. This class is recommended to load ready-to-work-with data.
    Expects input data as distributed on UCI's machine learning repository
    (either 'cup98LRN.txt' or 'cup98VAL.txt')."
    """

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
        'DOMAIN': ' ',
        'CHILD03': ' ',
        'CHILD07': ' ',
        'CHILD12': ' ',
        'CHILD18': ' ',
        'GEOCODE': ' ',
        'RFA_2': ' ',
        'RFA_3': ' ',
        'RFA_4': ' ',
        'RFA_5': ' ',
        'RFA_6': ' ',
        'RFA_7': ' ',
        'RFA_8': ' ',
        'RFA_9': ' ',
        'RFA_10': ' ',
        'RFA_11': ' ',
        'RFA_12': ' ',
        'RFA_13': ' ',
        'RFA_14': ' ',
        'RFA_15': ' ',
        'RFA_16': ' ',
        'RFA_17': ' ',
        'RFA_18': ' ',
        'RFA_19': ' ',
        'RFA_20': ' ',
        'RFA_21': ' ',
        'RFA_22': ' ',
        'RFA_23': ' ',
        'RFA_24': ' '}

    # Specs for boolean fields that need recoding
    recode_x_underscore = BooleanRecodeSpec({'true': 'X', 'false': '_'},
                                            ['NOEXCH',
                                             'MAJOR',
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
    recode_e_i = BooleanRecodeSpec({'true': "E", 'false': 'I'},['AGEFLAG'])
    recode_x_blank = BooleanRecodeSpec({'true': 'X', 'false': ' '},
                                       ['PEPSTRFL'])
    recode_h_u = BooleanRecodeSpec({'true': 'H', 'false': 'U'}, ['HOMEOWNR'])
    recode_b_blank = BooleanRecodeSpec({'true': 'B', 'false': ' '}, ['MAILCODE'])
    recode_1_0 = BooleanRecodeSpec({'true': '1', 'false': '0'}, ['TARGET_B','HPHONE_D'])

    boolean_recode_specs = [recode_x_underscore,
                      recode_y_n,
                      recode_e_i,
                      recode_x_blank,
                      recode_h_u,
                      recode_b_blank,
                      recode_1_0]

    # Fields with bytewise categorical data
    symbolic_fields = []
    symbolic_fields.append(SymbolicFieldSpreader(
                           "MDMAUD", ["Recency","Frequency","Amount"]))
    symbolic_fields.append(SymbolicFieldSpreader(
                           "DOMAIN",["Urbanicity", "SocioEconomicStatus"]))
    for i in range(2,25):
        field = "_".join(["RFA",str(i)])
        symbolic_fields.append(SymbolicFieldSpreader(
                               field, ["Recency","Frequency","Amount"]))

    # Some features of particular interest
    #######################################################################

    index_name = "CONTROLN"
    target_vars = ["TARGET_B", "TARGET_D"]

    date_features = ["ODATEDW","DOB","ADATE_2","ADATE_3","ADATE_4","ADATE_5","ADATE_6","ADATE_7","ADATE_8","ADATE_9","ADATE_10","ADATE_11","ADATE_12","ADATE_13","ADATE_14","ADATE_15","ADATE_16","ADATE_17","ADATE_18","ADATE_19","ADATE_20","ADATE_21","ADATE_22","ADATE_23","ADATE_24"]

    boolean_features = ["MAILCODE", "NOEXCH","RECINHSE","RECP3","RECPGVG","AGEFLAG","HOMEOWNR","MAJOR","COLLECT1","VETERANS","BIBLE","CATLG","HOMEE","PETS","CDPLAY","STEREO","PCOWNERS","PHOTO","CRAFTS","FISHER","GARDENIN","BOATS","WALKER","KIDSTUFF","CARDS","PLATES","REPSTRFL","TARGET_B","HPHONE_D"]

    categorical_features = ["TCODE", "STATE","PVASTATE","CLUSTER","CHILD03","CHILD07","CHILD12","CHILD18","GENDER","DATASRCE","SOLP3","SOLIH","WEALTH2","GEOCODE","LIFESRC","OSOURCE","RFA_2R","RFA_2A","MDMAUD_R", "MDMAUD_F","MDMAUD_A"]

    nominal_features = ["OSOURCE", "MDMAUD", "DOMAIN"]

    # Several features need special treatment on or after import. Where necessary, these are included here for explicit datatype casting
    # Categoricals are created unordered by default
    dtype_specs = {}
    for date in date_features:
        dtype_specs[date] = 'float'
    for boolean in boolean_features:
        dtype_specs[boolean] = 'str'
    for categorical in categorical_features:
        dtype_specs[categorical] = 'category'
    for nominal in nominal_features:
        dtype_specs[nominal] = 'str'

     # Donor title code. Pandas strips leading zeros on reading the data,
    # so this is reflected in the index keys here. (1 instead of 001 and so on)
    tcode_categories = {
        0: "_",
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

    # Features from US census
    us_census_features = ["POP901", "POP902", "POP903", "POP90C1", "POP90C2", "POP90C3", "POP90C4", "POP90C5", "ETH1", "ETH2", "ETH3", "ETH4", "ETH5", "ETH6", "ETH7", "ETH8", "ETH9", "ETH10", "ETH11", "ETH12", "ETH13", "ETH14", "ETH15", "ETH16", "AGE901", "AGE902", "AGE903", "AGE904", "AGE905", "AGE906", "AGE907", "CHIL1", "CHIL2", "CHIL3", "AGEC1", "AGEC2", "AGEC3", "AGEC4", "AGEC5", "AGEC6", "AGEC7", "CHILC1", "CHILC2", "CHILC3", "CHILC4", "CHILC5", "HHAGE1", "HHAGE2", "HHAGE3", "HHN1", "HHN2", "HHN3", "HHN4", "HHN5", "HHN6", "MARR1", "MARR2", "MARR3", "MARR4", "HHP1", "HHP2", "DW1", "DW2", "DW3", "DW4", "DW5", "DW6", "DW7", "DW8", "DW9", "HV1", "HV2", "HV3", "HV4", "HU1", "HU2", "HU3", "HU4", "HU5", "HHD1", "HHD2", "HHD3", "HHD4", "HHD5", "HHD6", "HHD7", "HHD8", "HHD9", "HHD10", "HHD11", "HHD12", "ETHC1", "ETHC2", "ETHC3", "ETHC4", "ETHC5", "ETHC6", "HVP1", "HVP2", "HVP3", "HVP4", "HVP5", "HVP6", "HUR1", "HUR2", "RHP1", "RHP2", "RHP3", "RHP4", "HUPA1", "HUPA2", "HUPA3", "HUPA4", "HUPA5", "HUPA6", "HUPA7", "RP1", "RP2", "RP3", "RP4", "MSA", "ADI", "DMA", "IC1", "IC2", "IC3", "IC4", "IC5", "IC6", "IC7", "IC8", "IC9", "IC10", "IC11", "IC12", "IC13", "IC14", "IC15", "IC16", "IC17", "IC18", "IC19", "IC20", "IC21", "IC22", "IC23", "HHAS1", "HHAS2", "HHAS3", "HHAS4", "MC1", "MC2", "MC3", "TPE1", "TPE2", "TPE3", "TPE4", "TPE5", "TPE6", "TPE7", "TPE8", "TPE9", "PEC1", "PEC2", "TPE10", "TPE11", "TPE12", "TPE13", "LFC1", "LFC2", "LFC3", "LFC4", "LFC5", "LFC6", "LFC7", "LFC8", "LFC9", "LFC10", "OCC1", "OCC2", "OCC3", "OCC4", "OCC5", "OCC6", "OCC7", "OCC8", "OCC9", "OCC10", "OCC11", "OCC12", "OCC13", "EIC1", "EIC2", "EIC3", "EIC4", "EIC5", "EIC6", "EIC7", "EIC8", "EIC9", "EIC10", "EIC11", "EIC12", "EIC13", "EIC14", "EIC15", "EIC16", "OEDC1", "OEDC2", "OEDC3", "OEDC4", "OEDC5", "OEDC6", "OEDC7", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7", "EC8", "SEC1", "SEC2", "SEC3", "SEC4", "SEC5", "AFC1", "AFC2", "AFC3", "AFC4", "AFC5", "AFC6", "VC1", "VC2", "VC3", "VC4", "ANC1", "ANC2", "ANC3", "ANC4", "ANC5", "ANC6", "ANC7", "ANC8", "ANC9", "ANC10", "ANC11", "ANC12", "ANC13", "ANC14", "ANC15", "POBC1", "POBC2", "LSC1", "LSC2", "LSC3", "LSC4", "VOC1", "VOC2", "VOC3", "HC1", "HC2", "HC3", "HC4", "HC5", "HC6", "HC7", "HC8", "HC9", "HC10", "HC11", "HC12", "HC13", "HC14", "HC15", "HC16", "HC17", "HC18", "HC19", "HC20", "HC21", "MHUC1", "MHUC2", "AC1", "AC2"]

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
        pull_stored: Whether to attempt loading tidy data from HDF store. Raw data is always pulled from the store if present.
        """
        self.hdf_data_file = App.config("hdf_store")
        self.pull_stored = pull_stored
        self.raw_data_file = csv_file
        self.hdf_store = App.config("data_dir")+App.config("hdf_store")
        self.raw_data = None
        self.processed_data = None

        if not csv_file is None and csv_file in [App.config("learn_file_name"),
                                                 App.config("validation_file_name")]:
            if "lrn" in csv_file.lower():
                self.dataset_type = App.config("learn_name")
                self.raw_dataset = App.config("learn_name_raw")
            elif "val" in csv_file.lower():
                self.dataset_type = App.config("validation_name")
                self.raw_dataset = App.config("validation_name_raw")
        else:
            raise NameError("Set csv_file to either training- or test-file.")

    def _rename_tcode(self):
        """
        Renames category labels for the donor title code.
        NOT USED! Extraneous levels in dataset.
        """
        if isinstance(self.processed_data, pd.DataFrame):
            # cast keys to string:
            new_cats = {str(k):str(v) for k,v in self.tcode_categories.items()}
            try:
                self.processed_data.TCODE.rename_categories(new_categories=new_cats,inplace=True)
            except ValueError as error:
                logger.error(error)

    def _format_zip(self, inplace=True):
        """
        Removes the dash at the end of some zip codes
        """
        zip_series = self.processed_data.ZIP.copy()
        zip_series = zip_series.str.replace('-','').astype('category')
        if inplace:
            self.processed_data.ZIP = zip_series
            return None
        else:
            return zip_series


    def _recode_booleans(self):
        """
        Recodes boolean columns. Specify the codes used in data and the
        affected columns through BooleanRecodeSpec objects.
        Expects a pandas data frame!
        """
        if not isinstance(self.processed_data, pd.DataFrame):
            raise TypeError("Needs a pandas dataframe.")

        if not all(isinstance(r, BooleanRecodeSpec) for r in self.boolean_recode_specs):
            raise TypeError("Expects a list of BooleanRecodeSpec objects.")

        for spec in self.boolean_recode_specs:
            spec.tds_ref = self # set reference to acess self.processed_data
            spec.do_recode()


    def _process_date_columns(self):
        """ Dates are stored as yymm in several columns. Split each of them into
        two new columns, containing the yy and mm parts separately. """

        for col in self.date_features:
            df = self.processed_data
            df[col] = df[col].astype(str)
            try:
                self.processed_data = df.join(df[col].str.extract(r'(?P<'+col+'_year>\d{2})(?P<'+col+'_month>\d{2})',
                             expand=True)).drop(col, axis=1)
                self.processed_data[col+'_year'].astype('category',copy=False)
                self.processed_data[col+'_month'].astype('category',copy=False)
            except Exception:
                logger.exception("Failed to convert date field %s", col)
                raise


    def _process_symbolic_fields(self):
        """ Processes the symbolic fields specified within this method.

        Params:
        -------
        data: A reference to TidyDataset.processed_data
        """
        if isinstance(self.processed_data, pd.DataFrame):
            # Call handler object's spreader method
            for f in self.symbolic_fields:
                f.set_tidy_dataset_ref(self)
                f.spread()
        else:
            raise NameError

    def _read_csv_data(self):
        """Read in csv data. After successful read, raw data is saved to HDF for future access."""
        try:
            logger.debug("trying to read"+self.get_raw_datafile_path())
            self.raw_data = pd.read_csv(
                self.get_raw_datafile_path(),
                index_col=self.index_name,
                na_values=self.na_codes,
                dtype=self.dtype_specs,
                low_memory=False,  # needed for mixed type columns
                memory_map=True  # load file in memory
                )
        except Exception as exc:
            logger.exception(exc)
            raise
        else:
            self._save_hdf(self.raw_dataset)

    def _process_raw(self):
        """
        Processes the raw csv import / raw data stored in HDF.
            - Recodes booleans
            - Splits multi-value columns (symbolic fields)
            - Renames categorical columns
        This method creates a copy of the raw data, which is preserved
        for later comparison.
        """
        if self.raw_data is None:
            self.get_raw_data()
        try:
            logger.debug("Processing"+self.raw_data_file)
            self.processed_data = self.raw_data.copy()
            self._format_zip()
            self._process_date_columns()
            self._recode_booleans()
            self._process_symbolic_fields()
        except Exception as error:
            self.processed_data = None
            logger.exception(error)
            raise

    def _load_hdf(self, key_name):
        """ Loads data from hdf store """

        # If data sould not be pulled, throw an exception to make sure
        # data is loaded from csv
        if not self.pull_stored:
            raise ValueError("HDF loading prohibited by options set.")
        if not key_name in [self.raw_dataset, self.dataset_type]:
            raise ValueError("Invalid HDF key. Cannot load data")

        try:
            if(key_name == self.raw_dataset):
                logger.debug("Loading "+self.raw_dataset+" from HDF.")
                self.raw_data = pd.read_hdf(self.get_hdf_datafile_path(),
                                            key=self.raw_dataset,
                                            mode='r')
            else:
                logger.debug("Loading "+self.dataset_type+" from HDF.")
                self.processed_data = pd.read_hdf(self.get_hdf_datafile_path(),
                                            key=self.dataset_type,
                                            mode='r')
        except (KeyError) as error:
            # If something goes wrong, pass the exception on to the caller
            logger.warning(error)
            raise error
        except(OSError, FileNotFoundError) as error:
            logger.info("HDF file not found. Will read from CSV.")
            raise error

    def _save_hdf(self, key_name):
        """ Save tidy data to hdf store """
        try:
            if key_name == self.dataset_type:
                self.processed_data.to_hdf(self.get_hdf_datafile_path(),
                                           key=key_name,
                                           format='table')
            elif key_name == self.raw_dataset:
                self.raw_data.to_hdf(self.get_hdf_datafile_path(),
                                     key=key_name,
                                     format='table')
        except Exception as exc:
            logger.error(exc)

    def _load_data(self):
        """
        Makes data available. First, checks if processed_data is alread present.
        If not, attempts to load from hdf, failing that, reads in csv and
        processes the data
        """
        if self.processed_data is None:
            try:
                self._load_hdf(self.dataset_type)
            except:
                try:
                    self.get_raw_data(inplace=True)
                except Exception as exc:
                    logger.error(exc)
                    raise
                else:
                    try:
                        self._process_raw()
                    except:
                        logger.error("Failed to process raw data.")
                        raise
                    else:
                        self._save_hdf(self.dataset_type)

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
            with columns TARGET_B and TARGET_D. Or None if the method is called on the validation dataset, which does not include these columns
        """

        if names_only:
            return pd.DataFrame(columns=App.config("dependent_vars"))

        if self.dataset_type == App.config("validation_name"):
            logger.debug("Tried to get dependent variables from validation dataset. Not found.")
            return None
        if self.processed_data is None:
            self._load_data()
        return self.processed_data.loc[:, App.config("dependent_vars")]


    def get_raw_data(self, inplace=False):
        if self.raw_data is None:
            try:
                self._load_hdf(self.raw_dataset)
            except(OSError, IOError, ValueError, KeyError):
                try:
                    self._read_csv_data()
                except Exception as exc:
                    logger.error(exc)
        if not inplace:
            return self.raw_data


    def get_tidy_data(self):
        """
        Gets processed data ready for further analysis. Attempts to load
        hdf, if that's not available, reads csv and processes data first.

        Returns
        -------
        processed_data: pandas.DataFrame object with tidy data
        """
        if self.processed_data is None:
            try:
                self._load_data()
            except Exception as error:
                logger.error(error)
            else:
                return self.processed_data
