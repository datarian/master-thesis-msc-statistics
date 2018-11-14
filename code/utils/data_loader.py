# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import os
import pandas as pd
import numpy as np
import logging
from config import App

# Set up the logger
logging.basicConfig(filename=__name__+'.log', level=logging.ERROR)
logger = logging.getLogger(__name__)


#######################################################################
# Main config
data_path = App.config("data_dir")
hdf_data_file_name = App.config("hdf_store")
hdf_store = os.path.join(App.config("data_dir"), App.config("hdf_store"))

#######################################################################
# Dicts and data structures to recode / reformat various variables
# and collections of related features

# Some features of particular interest
index_name = "CONTROLN"
labels = ["TARGET_B", "TARGET_D"]

drop_initial = ["MDMAUD", "RFA_2"]  # These are pre-split multibyte features
drop_redundant = ["FISTDATE", "NEXTDATE", "DOB"]

date_features = ["ODATEDW", "DOB", "ADATE_2", "ADATE_3", "ADATE_4",
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

binary_features = ["MAILCODE", "NOEXCH", "RECSWEEP", "RECINHSE", "RECP3",
                   "RECPGVG", "AGEFLAG", "HOMEOWNR", "MAJOR", "COLLECT1",
                   "BIBLE", "CATLG", "HOMEE", "PETS", "CDPLAY", "STEREO",
                   "PCOWNERS", "PHOTO", "CRAFTS", "FISHER", "GARDENIN",
                   "BOATS", "WALKER", "KIDSTUFF", "CARDS", "PLATES",
                   "PEPSTRFL", "TARGET_B", "HPHONE_D", "VETERANS"]

# Already usable nominal features
categorical_features = ["TCODE", "DOMAIN", "STATE", "PVASTATE", "CLUSTER", "INCOME",
                        "CHILD03", "CHILD07", "CHILD12", "CHILD18", "GENDER",
                        "DATASRCE", "SOLP3", "SOLIH", "WEALTH1", "WEALTH2",
                        "GEOCODE", "LIFESRC", "RFA_2R", "RFA_2A",
                        "RFA_2F", "MDMAUD_R", "MDMAUD_F", "MDMAUD_A",
                        "GEOCODE2","TARGET_D"]

# Nominal features needing further cleaning treatment
nominal_features = ["OSOURCE", "TCODE", "RFA_3", "RFA_4", "RFA_5", "RFA_6",
                    "RFA_7", "RFA_8", "RFA_9", "RFA_10", "RFA_11", "RFA_12",
                    "RFA_13", "RFA_14", "RFA_15", "RFA_16", "RFA_17", "RFA_18",
                    "RFA_19", "RFA_20", "RFA_21", "RFA_22", "RFA_23",
                    "RFA_24"]

us_census_features = ["POP901", "POP902", "POP903", "POP90C1", "POP90C2",
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

interest_features = ["COLLECT1", "VETERANS", "BIBLE", "CATLG", "HOMEE", "PETS",
                     "CDPLAY", "STEREO", "PCOWNERS", "PHOTO", "CRAFTS",
                     "FISHER", "GARDENIN", "BOATS", "WALKER", "KIDSTUFF",
                     "CARDS", "PLATES"]

promo_history_dates = ["ADATE_3", "ADATE_4", "ADATE_5", "ADATE_6",
                       "ADATE_7", "ADATE_8", "ADATE_9", "ADATE_10",
                       "ADATE_11", "ADATE_12", "ADATE_13",  "ADATE_14",
                       "ADATE_15", "ADATE_16", "ADATE_17", "ADATE_18",
                       "ADATE_19", "ADATE_20", "ADATE_21", "ADATE_22",
                       "ADATE_23", "ADATE_24"]

don_summary_dates = ["LASTDATE", "MINRDATE", "MAXRDATE", "MAXADATE"]

promo_history_summary = ['CARDPROM', 'MAXADATE', 'NUMPROM', 'CARDPM12',
                         'NUMPRM12']

giving_history_dates = ['RDATE_3', 'RDATE_4', 'RDATE_5', 'RDATE_6', 'RDATE_7',
                        'RDATE_8', 'RDATE_9', 'RDATE_10', 'RDATE_11',
                        'RDATE_12', 'RDATE_13', 'RDATE_14', 'RDATE_15',
                        'RDATE_16', 'RDATE_17', 'RDATE_18', 'RDATE_19',
                        'RDATE_20', 'RDATE_21', 'RDATE_22', 'RDATE_23',
                        'RDATE_24']

giving_history = ['RAMNT_3', 'RAMNT_4', 'RAMNT_5', 'RAMNT_6',
                  'RAMNT_7', 'RAMNT_8', 'RAMNT_9', 'RAMNT_10', 'RAMNT_11',
                  'RAMNT_12', 'RAMNT_13', 'RAMNT_14', 'RAMNT_15',
                  'RAMNT_16', 'RAMNT_17', 'RAMNT_18', 'RAMNT_19',
                  'RAMNT_20', 'RAMNT_21', 'RAMNT_22', 'RAMNT_23',
                  'RAMNT_24']

giving_history_summary = ['RAMNTALL', 'NGIFTALL', 'MINRAMNT', 'MAXRAMNT',
                          'LASTGIFT', 'TIMELAG', 'AVGGIFT']

# Explicitly define NA codes globally
# The codes are specified in the dataset documentation.
na_codes = ['', '.', ' ']


def dateparser(date_features):

    reference_date = App.config("reference_date")

    def fix_format(d):
        if not pd.isna(d):
            if len(d) == 3:
                d = '0'+d
        else:
            d = pd.NaT
        return d

    def fix_century(d):
        ref_date = App.config("reference_date")
        if not pd.isna(d):
            try:
                if d.year > ref_date.year:
                    d = d.replace(year=(d.year-100))
            except Exception as err:
                logger.warning("Failed to fix century for date {}, reason: {}".format(d, err))
                d = pd.NaT
        else:
            d = pd.NaT
        return d

    try:
        date_features = [fix_century(pd.to_datetime(fix_format(d), format="%y%m", errors="coerce")) for d in date_features]
    except Exception as e:
        logger.warn("Failed to parse date array {}.\nReason: {}".format(date_features, e))
    return date_features


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
    for binary in binary_features:
        dtype_specs[binary] = 'str'
    for categorical in categorical_features:
        dtype_specs[categorical] = 'category'
    for nominal in nominal_features:
        dtype_specs[nominal] = 'str'
    for date in date_features:
        dtype_specs[date] = 'str'

    def __init__(self, csv_file=None, pull_stored=True):
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
        self.pull_stored = pull_stored
        self.raw_data = None

        self.reference_date = App.config("reference_date")

        if csv_file is not None and csv_file in [App.config("learn_file_name"),
                                                 App.config("validation_file_name")]:
            if "lrn" in csv_file.lower():
                self.raw_dataset = App.config("learn_name")
            elif "val" in csv_file.lower():
                self.raw_dataset = App.config("validation_name")
        else:
            raise NameError("Set csv_file to either training- or test-file.")

    def _read_csv_data(self):
        """
        Read in csv data. After successful read,
        raw data is saved to HDF for future access. A few fetures
        are already dropped at this stage (see: data_loader.drop_initial)!
        """

        try:
            logger.debug("trying to read csv file: "+self.raw_data_file_name)
            self.raw_data = pd.read_csv(
                os.path.join(data_path, self.raw_data_file_name),
                index_col=index_name,
                na_values=na_codes,
                parse_dates=date_features,
                date_parser=dateparser,
                dtype=self.dtype_specs,
                low_memory=False,  # needed for mixed type columns
                memory_map=True  # load file in memory
            )

            # Fix formatting for ZIP feature
            self.raw_data.ZIP = self.raw_data.ZIP.str.replace('-', '').replace([' ', '.'], np.nan).astype('float64')
            # Fix binary encoding inconsistency for NOEXCH
            self.raw_data.NOEXCH = self.raw_data.NOEXCH.str.replace("X", "1")

            # Fix some NA value problems:
            self.raw_data[['MDMAUD_R', 'MDMAUD_F', 'MDMAUD_A']] = self.raw_data.loc[:,['MDMAUD_R', 'MDMAUD_F', 'MDMAUD_A']].replace('X', np.nan)

            # Drop obivously redundant features
            self.raw_data = self.raw_data.drop(drop_initial, axis=1)

            
        except Exception as exc:
            logger.exception(exc)
            raise
        else:
            self._save_hdf(self.raw_dataset)

    def _load_hdf(self, key_name):
        """ Loads data from hdf store """

        # If data should not be pulled, throw an exception to make sure
        # data is loaded from csv
        if not self.pull_stored:
            raise ValueError("HDF loading prohibited by options set.")
        try:
            logger.debug("Loading "+self.raw_dataset+" from HDF.")
            self.raw_data = pd.read_hdf(hdf_store,
                                        key=self.raw_dataset,
                                        mode='r')
        except (KeyError) as error:
            # If something goes wrong, pass the exception on to the caller
            logger.warning(error)
            raise error
        except(OSError, FileNotFoundError) as error:
            logger.info("HDF file not found. Will read from CSV.")
            raise error

    def _save_hdf(self, key_name):
        """ Save Pandas dataframe to hdf store"""
        try:
            self.raw_data.to_hdf(hdf_store,
                                 key=key_name,
                                 format='table')
        except Exception as exc:
            logger.error(exc)
            raise exc

    def get_dataset(self):
        """
        Returns the raw dataset without any transformations applied,
        as read from the csv file.
        """
        if self.raw_data is None:
            try:
                self._load_hdf(self.raw_dataset)
            except(OSError, IOError, ValueError, KeyError) as exc:
                # The hdf file is not there, or nothing saved under
                # the key we tried to query.
                try:
                    self._read_csv_data()
                except Exception as exc:
                    logger.error(exc)
                    raise exc
        return self.raw_data.copy()
