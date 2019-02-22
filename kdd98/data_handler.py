# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import logging
import os
import pathlib
import urllib
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import kdd98.utils_transformer as ut
from kdd98.config import Config
from kdd98.transformers import (BinaryFeatureRecode, DeltaTime,
                                MonthsToDonation, MultiByteExtract,
                                OrdinalEncoder, RecodeUrbanSocioEconomic)

# Set up the logger
logging.basicConfig(filename=__name__+'.log', level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    'KDD98DataLoader',
    'Cleaner',
    'INDEX_NAME',
    'TARGETS',
    'DATE_FEATURES',
    'PROMO_HISTORY_DATES',
    'BINARY_FEATURES',
    'CATEGORICAL_FEATURES',
    'NOMINAL_FEATURES',
    'ORDINAL_MAPPING_MDMAUD',
    'ORDINAL_MAPPING_RFA',
    'INTEREST_FEATURES',
    'DON_SUMMARY_DATES',
    'PROMO_HISTORY_SUMMARY',
    'GIVING_HISTORY_DATES',
    'GIVING_HISTORY',
    'GIVING_HISTORY_SUMMARY'
]


#######################################################################
# Main config
data_path = Config.get("data_dir")
hdf_data_file_name = Config.get("hdf_store")
hdf_store = pathlib.Path(data_path.resolve(), hdf_data_file_name)

#######################################################################
# Dicts and data structures to recode / reformat various variables
# and collections of related features

# Some features of particular interest
INDEX_NAME = "CONTROLN"
TARGETS = ["TARGET_B", "TARGET_D"]

DROP_INITIAL = ["MDMAUD", "RFA_2"]  # These are pre-split multibyte features
# These are contained in other features
DROP_REDUNDANT = ["FISTDATE", "NEXTDATE", "DOB"]

DATE_FEATURES = ["ODATEDW", "DOB", "ADATE_2", "ADATE_3", "ADATE_4",
                 "ADATE_5", "ADATE_6", "ADATE_7", "ADATE_8", "ADATE_9",
                 "ADATE_10", "ADATE_11", "ADATE_12", "ADATE_13",
                 "ADATE_14", "ADATE_15", "ADATE_16", "ADATE_17",
                 "ADATE_18", "ADATE_19", "ADATE_20", "ADATE_21",
                 "ADATE_22", "ADATE_23", "ADATE_24",
                 "RDATE_3", "RDATE_4", "RDATE_5", "RDATE_6",
                 "RDATE_7", "RDATE_8", "RDATE_9", "RDATE_10",
                 "RDATE_11", "RDATE_12", "RDATE_13", "RDATE_14",
                 "RDATE_15", "RDATE_16", "RDATE_17", "RDATE_18",
                 "RDATE_19", "RDATE_20", "RDATE_21", "RDATE_22",
                 "RDATE_23", "RDATE_24", "LASTDATE", "MINRDATE",
                 "MAXRDATE", "FISTDATE", "NEXTDATE", "MAXADATE"]

BINARY_FEATURES = ["MAILCODE", "NOEXCH", "RECSWEEP", "RECINHSE", "RECP3",
                   "RECPGVG", "AGEFLAG", "HOMEOWNR", "MAJOR", "COLLECT1",
                   "BIBLE", "CATLG", "HOMEE", "PETS", "CDPLAY", "STEREO",
                   "PCOWNERS", "PHOTO", "CRAFTS", "FISHER", "GARDENIN",
                   "BOATS", "WALKER", "KIDSTUFF", "CARDS", "PLATES",
                   "PEPSTRFL", "TARGET_B", "HPHONE_D", "VETERANS"]

# Already usable nominal features
CATEGORICAL_FEATURES = ["OSOURCE", "TCODE", "DOMAIN", "STATE", "PVASTATE", "CLUSTER", "INCOME",
                        "CHILD03", "CHILD07", "CHILD12", "CHILD18", "GENDER",
                        "DATASRCE", "SOLP3", "SOLIH", "WEALTH1", "WEALTH2",
                        "GEOCODE", "LIFESRC", "RFA_2R", "RFA_2A",
                        "RFA_2F", "MDMAUD_R", "MDMAUD_F", "MDMAUD_A",
                        "GEOCODE2", "TARGET_D"]

# Nominal features needing further cleaning treatment
NOMINAL_FEATURES = ["OSOURCE", "TCODE", "RFA_3", "RFA_4", "RFA_5", "RFA_6",
                    "RFA_7", "RFA_8", "RFA_9", "RFA_10", "RFA_11", "RFA_12",
                    "RFA_13", "RFA_14", "RFA_15", "RFA_16", "RFA_17", "RFA_18",
                    "RFA_19", "RFA_20", "RFA_21", "RFA_22", "RFA_23",
                    "RFA_24"]

ORDINAL_MAPPING_MDMAUD = [
    {'col': 'MDMAUD_R', 'mapping': {'D': 1, 'I': 2, 'L': 3, 'C': 4}},
    {'col': 'MDMAUD_A', 'mapping': {'L': 1, 'C': 2, 'M': 3, 'T': 4}}]

ORDINAL_MAPPING_RFA = [{'col': c, 'mapping': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}}
                       for c in ['RFA_2A', 'RFA_3A', 'RFA_4A', 'RFA_5A',
                                 'RFA_6A', 'RFA_7A', 'RFA_8A', 'RFA_9A',
                                 'RFA_10A', 'RFA_11A', 'RFA_12A', 'RFA_13A',
                                 'RFA_14A', 'RFA_15A', 'RFA_16A', 'RFA_17A',
                                 'RFA_18A', 'RFA_19A', 'RFA_20A', 'RFA_21A',
                                 'RFA_22A', 'RFA_23A', 'RFA_24A']]

US_CENSUS_FEATURES = ["POP901", "POP902", "POP903", "POP90C1", "POP90C2",
                      "POP90C3", "POP90C4", "POP90C5", "ETH1", "ETH2",
                      "ETH3", "ETH4", "ETH5", "ETH6", "ETH7", "ETH8",
                      "ETH9", "ETH10", "ETH11", "ETH12", "ETH13", "ETH14",
                      "ETH15", "ETH16", "AGE901", "AGE902", "AGE903",
                      "AGE904", "AGE905", "AGE906", "AGE907", "CHIL1",
                      "CHIL2", "CHIL3", "AGEC1", "AGEC2", "AGEC3", "AGEC4",
                      "AGEC5", "AGEC6", "AGEC7", "CHILC1", "CHILC2",
                      "CHILC3", "CHILC4", "CHILC5", "HHAGE1", "HHAGE2",
                      "HHAGE3", "HHN1", "HHN2", "HHN3", "HHN4", "HHN5",
                      "HHN6", "MARR1", "MARR2", "MARR3", "MARR4", "HHP1",
                      "HHP2", "DW1", "DW2", "DW3", "DW4", "DW5", "DW6",
                      "DW7", "DW8", "DW9", "HV1", "HV2", "HV3", "HV4",
                      "HU1", "HU2", "HU3", "HU4", "HU5", "HHD1", "HHD2",
                      "HHD3", "HHD4", "HHD5", "HHD6", "HHD7", "HHD8",
                      "HHD9", "HHD10", "HHD11", "HHD12", "ETHC1", "ETHC2",
                      "ETHC3", "ETHC4", "ETHC5", "ETHC6", "HVP1", "HVP2",
                      "HVP3", "HVP4", "HVP5", "HVP6", "HUR1", "HUR2",
                      "RHP1", "RHP2", "RHP3", "RHP4", "HUPA1", "HUPA2",
                      "HUPA3", "HUPA4", "HUPA5", "HUPA6", "HUPA7", "RP1",
                      "RP2", "RP3", "RP4", "MSA", "ADI", "DMA", "IC1",
                      "IC2", "IC3", "IC4", "IC5", "IC6", "IC7", "IC8",
                      "IC9", "IC10", "IC11", "IC12", "IC13", "IC14",
                      "IC15", "IC16", "IC17", "IC18", "IC19", "IC20",
                      "IC21", "IC22", "IC23", "HHAS1", "HHAS2", "HHAS3",
                      "HHAS4", "MC1", "MC2", "MC3", "TPE1", "TPE2", "TPE3",
                      "TPE4", "TPE5", "TPE6", "TPE7", "TPE8", "TPE9",
                      "PEC1", "PEC2", "TPE10", "TPE11", "TPE12", "TPE13",
                      "LFC1", "LFC2", "LFC3", "LFC4", "LFC5", "LFC6",
                      "LFC7", "LFC8", "LFC9", "LFC10", "OCC1", "OCC2",
                      "OCC3", "OCC4", "OCC5", "OCC6", "OCC7", "OCC8",
                      "OCC9", "OCC10", "OCC11", "OCC12", "OCC13", "EIC1",
                      "EIC2", "EIC3", "EIC4", "EIC5", "EIC6", "EIC7",
                      "EIC8", "EIC9", "EIC10", "EIC11", "EIC12", "EIC13",
                      "EIC14", "EIC15", "EIC16", "OEDC1", "OEDC2", "OEDC3",
                      "OEDC4", "OEDC5", "OEDC6", "OEDC7", "EC1", "EC2",
                      "EC3", "EC4", "EC5", "EC6", "EC7", "EC8", "SEC1",
                      "SEC2", "SEC3", "SEC4", "SEC5", "AFC1", "AFC2",
                      "AFC3", "AFC4", "AFC5", "AFC6", "VC1", "VC2", "VC3",
                      "VC4", "ANC1", "ANC2", "ANC3", "ANC4", "ANC5",
                      "ANC6", "ANC7", "ANC8", "ANC9", "ANC10", "ANC11",
                      "ANC12", "ANC13", "ANC14", "ANC15", "POBC1", "POBC2",
                      "LSC1", "LSC2", "LSC3", "LSC4", "VOC1", "VOC2",
                      "VOC3", "HC1", "HC2", "HC3", "HC4", "HC5", "HC6",
                      "HC7", "HC8", "HC9", "HC10", "HC11", "HC12", "HC13",
                      "HC14", "HC15", "HC16", "HC17", "HC18", "HC19",
                      "HC20", "HC21", "MHUC1", "MHUC2", "AC1", "AC2"]

INTEREST_FEATURES = ["COLLECT1", "VETERANS", "BIBLE", "CATLG", "HOMEE", "PETS",
                     "CDPLAY", "STEREO", "PCOWNERS", "PHOTO", "CRAFTS",
                     "FISHER", "GARDENIN", "BOATS", "WALKER", "KIDSTUFF",
                     "CARDS", "PLATES"]

PROMO_HISTORY_DATES = ["ADATE_3", "ADATE_4", "ADATE_5", "ADATE_6",
                       "ADATE_7", "ADATE_8", "ADATE_9", "ADATE_10",
                       "ADATE_11", "ADATE_12", "ADATE_13",  "ADATE_14",
                       "ADATE_15", "ADATE_16", "ADATE_17", "ADATE_18",
                       "ADATE_19", "ADATE_20", "ADATE_21", "ADATE_22",
                       "ADATE_23", "ADATE_24"]

DON_SUMMARY_DATES = ["LASTDATE", "MINRDATE", "MAXRDATE", "MAXADATE"]

PROMO_HISTORY_SUMMARY = ['CARDPROM', 'MAXADATE', 'NUMPROM', 'CARDPM12',
                         'NUMPRM12']

GIVING_HISTORY_DATES = ['RDATE_3', 'RDATE_4', 'RDATE_5', 'RDATE_6', 'RDATE_7',
                        'RDATE_8', 'RDATE_9', 'RDATE_10', 'RDATE_11',
                        'RDATE_12', 'RDATE_13', 'RDATE_14', 'RDATE_15',
                        'RDATE_16', 'RDATE_17', 'RDATE_18', 'RDATE_19',
                        'RDATE_20', 'RDATE_21', 'RDATE_22', 'RDATE_23',
                        'RDATE_24']

GIVING_HISTORY = ['RAMNT_3', 'RAMNT_4', 'RAMNT_5', 'RAMNT_6',
                  'RAMNT_7', 'RAMNT_8', 'RAMNT_9', 'RAMNT_10', 'RAMNT_11',
                  'RAMNT_12', 'RAMNT_13', 'RAMNT_14', 'RAMNT_15',
                  'RAMNT_16', 'RAMNT_17', 'RAMNT_18', 'RAMNT_19',
                  'RAMNT_20', 'RAMNT_21', 'RAMNT_22', 'RAMNT_23',
                  'RAMNT_24']

GIVING_HISTORY_SUMMARY = ['RAMNTALL', 'NGIFTALL', 'MINRAMNT', 'MAXRAMNT',
                          'LASTGIFT', 'TIMELAG', 'AVGGIFT']


# Explicitly define NA codes globally
# The codes are specified in the dataset documentation.
NA_CODES = ['', '.']


class KDD98DataLoader:
    """ Handles dataset loading and stores datasets in hdf store
    for speedy future access.
    Expects input data as distributed on UCI's machine learning repository
    (either 'cup98LRN.txt' or 'cup98VAL.txt').
    """

    # Several features need special treatment on import.
    # Where necessary, these are included here for explicit datatype casting.
    # The rest of the features will be guessed by pandas on reading the CSV.
    dtype_specs = {}
    for binary in BINARY_FEATURES:
        dtype_specs[binary] = 'str'
    for categorical in CATEGORICAL_FEATURES:
        dtype_specs[categorical] = 'category'
    for nominal in NOMINAL_FEATURES:
        dtype_specs[nominal] = 'str'
    for date in DATE_FEATURES:
        dtype_specs[date] = 'str'

    def __init__(self, csv_file=None, pull_stored=True, download_url=None):
        """
        Initializes a new object. No data is pulled rightaway. Magically
        initializes certain features based on the occurrence of either of the
        strings ['LRN','VAL'] in the filename. So please do not rename the
        files (as distributed on the UCI repository arbitrarily.

        Parameters:
        -----------
        csv_file: The name of the training or test file (cup98[LRN,VAL].txt)
        pull_stored: Whether to attempt loading raw data from HDF store.
        """
        self.raw_data_file_name = csv_file
        self.raw_data_name = None
        self.clean_data_name = None
        self._raw_data = pd.DataFrame()
        self._clean_data = pd.DataFrame()
        self._preprocessed_data = pd.DataFrame()

        self.download_url = download_url
        self.reference_date = Config.get("reference_date")

        if csv_file is not None and csv_file in Config.get("learn_file_name", "learn_test_file_name", "validation_file_name"):
            self.raw_data_name = pathlib.Path(csv_file).stem # new
            logger.info("Set raw data file name to: {:1}".format(self.raw_data_name))
            if "lrn" in csv_file.lower():
                self.clean_data_name = Config.get("learn_clean_name")
                self.preproc_data_name = Config.get("learn_preproc_name")
            elif "val" in csv_file.lower():
                self.clean_data_name = Config.get("validation_clean_name")
                self.preproc_data_name = Config.get("validation_preproc_name")
        else:
            raise ValueError("Set csv_file to either training- or test-file.")

    @property
    def raw_data(self):
        if self._raw_data.empty:
            self.provide("raw")
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value):
        self._raw_data = value

    @property
    def clean_data(self):
        if self._clean_data.empty:
            self.provide("clean")
        return self._clean_data

    @clean_data.setter
    def clean_data(self, value):
        self._clean_data = value

    @property
    def preprocessed_data(self):
        if self._preprocessed_data.empty:
            self.provide("preproc")
        return self._preprocessed_data

    @preprocessed_data.setter
    def preprocessed_data(self, value):
        self._preprocessed_data = value

    def provide(self, type):
        """
        Provides data by first checking the hdf store, then loading csv data.

        If clean data is requested, the returned pandas object has:
        - binary
        - numerical (float, int)
        - ordinal / nominal categorical
        - all missing values np.nan
        - dates in np.datetime64

        If preprocessed data is requested, the returned pandas object has
        - the contents of cleaned data
        - date features transformed to time deltas
        - encoded categoricals

        data in it.

        Params
        ------
        type    One of ["raw", "clean", "preproc"]. Raw is as read by pandas, clean is with cleaning operations applied.
        """
        name_mapper = {
            "raw": {"key": self.raw_data_name,
                    "data_attrib": "_raw_data"},
            "clean": {"key": self.clean_data_name,
                      "data_attrib": "_clean_data"},
            "preproc": {"key": self.preproc_data_name,
                        "data_attrib": "_preprocessed_data"}
        }

        assert(type in ["raw", "clean", "preproc"])

        try:
            # First, try to load the data from hdf
            # and set the object
            data = self._load_hdf(name_mapper[type]["key"])
            setattr(self, name_mapper[type]["data_attrib"], data)
        except:
            # If it fails and we ask for clean data,
            # try to find the raw data in hdf and, if present,
            # load it. If we ask for preprocessed data, try to find
            # cleaned data in hdf and load if present.
            if type == "clean":
                try:
                    self.provide("raw")
                except Exception as e:
                    logger.error("Failed to provide raw data. Cannot provide clean data.\nReason: {:1}".format(e))
                try:
                    cln = Cleaner(self)
                    self.clean_data = cln.clean()
                except Exception as e:
                    logger.error("Failed to clean raw data.\nReason: {}".format(e))
                    raise e
                self._save_hdf(self.clean_data, self.clean_data_name)
            elif type == "preproc":
                try:
                    self.provide("clean")
                except Exception as e:
                    logger.error("Failed to provide clean data. Cannot provide preprocessed data.\nReason: {}".format(e))
                try:
                    pre = Cleaner(self)
                    self.preprocessed_data = pre.preprocess()
                except Exception as e:
                    logger.error("Failed to preprocess clean data.\nReason: {}".format(e))
                    raise e
                self._save_hdf(self.preprocessed_data, self.preproc_data_name)
            else:
                try:
                    self._read_csv_data()
                except Exception as error:
                    logger.error(
                        "Failed to load data from csv file {:1}!".format(self.raw_data_file_name))
                    raise error

    def _read_csv_data(self):
        """
        Read in csv data. After successful read, raw data is saved to HDF for future access.
        """

        try:
            data_file = Config.get("data_dir") / self.raw_data_file_name
            if not data_file.is_file():
                logger.info("Data not stored locally. Downloading...")
                try:
                    self._fetch_online(self.download_url)
                except urllib.error.HTTPError:
                    logger.error(
                        "Failed to download dataset from: {}.".format(self.download_url))

            logger.info("Reading csv file: "+self.raw_data_file_name)
            self.raw_data = pd.read_csv(
                pathlib.Path(Config.get("data_dir"), self.raw_data_file_name),
                index_col=INDEX_NAME,
                na_values=NA_CODES,
                dtype=self.dtype_specs,
                low_memory=False,  # needed for mixed type columns
                memory_map=True  # load file in memory
            )
        except Exception as exc:
            logger.error(exc)
            raise
        self._save_hdf(self.raw_data, self.raw_data_name)

    def _load_hdf(self, key_name):
        """ Loads data from hdf store.
        Raises an error if the key or the file is not found.

        Params
        ------
        key_name    The key to load

        """

        try:
            store = pd.HDFStore(hdf_store, mode="r")
            logger.info("Trying to load {:1} from HDF.".format(key_name))
            dataset = pd.read_hdf(store,
                               key=key_name)
        except (KeyError) as error:
            # If something goes wrong, pass the exception on to the caller
            logger.info("Key not found in HDF store. Reading from CSV.")
            raise error
        except(OSError, FileNotFoundError) as error:
            logger.info("HDF file not found. Will read from CSV.")
            raise error
        finally:
            store.close()
        return dataset

    def _save_hdf(self, data, key_name):
        """ Save a pandas dataframe to hdf store. The hdf format 'table' is
        used, which is slower but supports pandas data types. Theoretically,
        it also allows to query the object and return subsets.

        Params
        ------
        data    A pandas dataframe or other object
        key_name    The key name to store the object at.
        """
        try:
            store = pd.HDFStore(hdf_store, mode="a")
            data.to_hdf(store,
                        key=key_name,
                        format="table",
                        mode="a")
        except Exception as exc:
            logger.error(exc)
            raise exc
        finally:
            store.close()

    def _fetch_online(self, url=None, dl_dir=None):
        """
        Fetches the data from the specified url or from the UCI machine learning database.

        Params:
        url:    Optional url to fetch from. Default is UCI machine learning database.
        """

        if not url:
            url = Config.get("download_url")

        if dl_dir:
            path = pathlib.Path(dl_dir)
        else:
            path = pathlib.Path(Config.get("data_dir"))
        contents = [f for f in path.iterdir()]
        missing = set(Config.get('download_files')) - set(contents)
        print("Files missing: {}".format(missing))
        if missing:
            for f in missing:
                file = path / f
                urllib.request.urlretrieve(url+'/'+f, file)
                if(pathlib.Path(f).suffix == '.zip'):
                    with zipfile.ZipFile(file, mode='r') as archive:
                        archive.extractall(path=path)


class Cleaner:

    def __init__(self, data_loader):
        self.dl = data_loader
        assert(self.dl.raw_data_file_name in Config.get("learn_file_name", "learn_test_file_name", "validation_file_name"))
        self.dimension_cols = None

    def drop_if_exists(self, data, features):

        for f in features:
            try:
                data.drop(f, axis=1, inplace=True)
            except KeyError:
                logger.info("Tried dropping feature {}, but it was not present in the data. Possibly alreay removed earlier.".format(f))
        return data


    def clean(self):
        data = self.dl.raw_data.copy(deep=True)
        logger.info("Starting cleaning of raw dataset")

        # Transforming the data and rebuilding the pandas dataframe
        drop_features = set()
        # Some features are redundant, these are removed here
        drop_features.update(DROP_INITIAL)
        drop_features.update(DROP_REDUNDANT)

        # Fix input errors for date features
        # The parser used on the date features
        def fix_format(d):
            if not pd.isna(d):
                if len(d) == 3:
                    d = "0"+d
            return d

        data[DATE_FEATURES] = data.loc[:,DATE_FEATURES].applymap(fix_format)

        # Fix formatting for ZIP feature
        data.ZIP = data.ZIP.str.replace(
            "-", "").replace([" ", "."], np.nan).astype("int64")
        # Fix binary encoding inconsistency for NOEXCH
        data.NOEXCH = data.NOEXCH.str.replace("X", "1")
        # Fix some NA value problems:
        data[["MDMAUD_R", "MDMAUD_F", "MDMAUD_A"]] = data.loc[:, ["MDMAUD_R", "MDMAUD_F", "MDMAUD_A"]].replace("X", np.nan)

        # Binary Features
        binary_transformers = ColumnTransformer([
            ("binary_x_bl",
             BinaryFeatureRecode(
                 value_map={"true": "X", "false": " "}, correct_noisy=False),
             ["PEPSTRFL", "MAJOR", "RECINHSE",
                 "RECP3", "RECPGVG", "RECSWEEP"]
             ),
            ("binary_y_n",
             BinaryFeatureRecode(
                 value_map={"true": "Y", "false": "N"}, correct_noisy=False),
             ["COLLECT1", "VETERANS", "BIBLE", "CATLG", "HOMEE", "PETS", "CDPLAY", "STEREO",
              "PCOWNERS", "PHOTO", "CRAFTS", "FISHER", "GARDENIN", "BOATS", "WALKER", "KIDSTUFF",
              "CARDS", "PLATES"]
             ),
            ("binary_e_i",
             BinaryFeatureRecode(
                 value_map={"true": "E", "false": "I"}, correct_noisy=False),
             ["AGEFLAG"]
             ),
            ("binary_h_u",
             BinaryFeatureRecode(
                 value_map={"true": "H", "false": "U"}, correct_noisy=False),
             ["HOMEOWNR"]),
            ("binary_b_bl",
             BinaryFeatureRecode(
                 value_map={"true": "B", "false": " "}, correct_noisy=False),
             ["MAILCODE"]
             ),
            ("binary_1_0",
             BinaryFeatureRecode(
                 value_map={"true": "1", "false": "0"}, correct_noisy=False),
             ["HPHONE_D", "NOEXCH"]
             )
        ])
        binarys = binary_transformers.fit_transform(data)
        data = ut.update_df_with_transformed(
            data, binarys, binary_transformers)

        # Multibyte Categoricals
        multibyte_transformer = ColumnTransformer([
            ("spread_rfa",
             MultiByteExtract(["R", "F", "A"]),
             NOMINAL_FEATURES[2:]),
             ("spread_domain",
             MultiByteExtract(["Urbanicity", "SocioEconomic"]),
             ["DOMAIN"])
        ])
        multibytes = multibyte_transformer.fit_transform(data)
        # The original multibyte-features are dropped at a later stage
        drop_features.update(NOMINAL_FEATURES[2:])
        drop_features.update(["DOMAIN"])

        data = ut.update_df_with_transformed(
            data, multibytes, multibyte_transformer, new_dtype="category")

        # Ordinals
        # This transformation MUST be defined and applied after splitting
        # multibyte features!
        ordinal_transformer = ColumnTransformer([
            ("order_mdmaud",
             OrdinalEncoder(mapping=ORDINAL_MAPPING_MDMAUD,
                            handle_unknown="ignore"),
             ["MDMAUD_R", "MDMAUD_A"]),
            ("order_rfa",
             OrdinalEncoder(mapping=ORDINAL_MAPPING_RFA,
                            handle_unknown="ignore"),
                            list(data.filter(regex=r"RFA_\d{1,2}A", axis=1).columns.values)),
            ("recode_socioecon", RecodeUrbanSocioEconomic(), ["DOMAINUrbanicity", "DOMAINSocioEconomic"])
        ])
        ordinals = ordinal_transformer.fit_transform(data)
        data = ut.update_df_with_transformed(
            data, ordinals, ordinal_transformer, new_dtype="category")

        # Ensure the remaining, already numeric ordinal features are in the correct pandas data type
        remaining_ordinals = ["WEALTH1","WEALTH2","INCOME"]+data.filter(
            regex=r"RFA_\d{1,2}F").columns.values.tolist()

        for f in data[remaining_ordinals]:
            try:
                data[f] = data[f].cat.as_ordered()
            except AttributeError:
                data[f] = data[f].astype("category").cat.as_ordered()

        # Now, drop all features marked for removal
        logger.info("About to drop these features in cleaning: {}".format(drop_features))
        data = self.drop_if_exists(data, drop_features)

        remaining_object_features = data.select_dtypes(include="object").columns.values.tolist()
        remaining_without_dates = [r for r in remaining_object_features if r not in DATE_FEATURES]

        if remaining_without_dates:
            logger.warning("After cleaning, the following features were left untreated and automatically coerced to 'category' (nominal): {}".format(remaining_without_dates))
            data[remaining_without_dates] = data[remaining_without_dates].astype("category")
        logger.info("Cleaning completed...")
        return data

    def preprocess(self):

        data = self.dl.clean_data.copy(deep=True)
        logger.info("Starting preprocessing of clean dataset")

        drop_features = set()

        # Treat date features. These are converted to time deltas
        # in either months or years
        don_hist_transformer = ColumnTransformer([
            ("months_to_donation",
            MonthsToDonation(),
            PROMO_HISTORY_DATES+GIVING_HISTORY_DATES
            )
        ])
        donation_responses = don_hist_transformer.fit_transform(data)
        data = ut.update_df_with_transformed(
            data, donation_responses, don_hist_transformer)

        logger.info("About to drop these features in preprocessing: {}".format(drop_features))
        drop_features.update(PROMO_HISTORY_DATES+GIVING_HISTORY_DATES)

        timedelta_transformer = ColumnTransformer([
            ("time_last_donation", DeltaTime(unit="months"), ["LASTDATE","MINRDATE","MAXRDATE","MAXADATE"]),
            ("membership_years", DeltaTime(unit="years"),["ODATEDW"])
        ])
        timedeltas = timedelta_transformer.fit_transform(data)
        data = ut.update_df_with_transformed(data, timedeltas, timedelta_transformer)

        drop_features.update(DATE_FEATURES)


        data = self.drop_if_exists(data, drop_features)

        # Imputation

        # For the donation amounts per campaign, NaN actually means 0 dollars donated, so change this accordingly.
        data[GIVING_HISTORY] = data.loc[:,GIVING_HISTORY].fillna(0, axis=1)

        logger.info("Preprocessing completed...")
        return data

class Engineer:
    pass
