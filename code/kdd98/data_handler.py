# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import logging
import pathlib
import pickle as pkl
import urllib
import zipfile
from collections import OrderedDict

import kdd98.utils_transformer as ut
import pandas as pd
from category_encoders import BinaryEncoder, OneHotEncoder
from kdd98.config import Config
from kdd98.transformers import (AllRelevantFeatureFilter, BinaryFeatureRecode,
                                CategoricalImputer, DateFormatter, DeltaTime,
                                MDMAUDFormatter, MedianImputer,
                                MonthsToDonation, MultiByteExtract,
                                NOEXCHFormatter, OrdinalEncoder, RAMNTFixer,
                                RFAFixer, ZeroVarianceSparseDropper,
                                ZipFormatter, ZipToCoords)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Set up the logger
logging.basicConfig(filename=__name__ + '.log', level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    'KDD98DataProvider',
    'Preprocessor',
    'Engineer',
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
                   "PEPSTRFL", "HPHONE_D", "VETERANS"]

# Already usable nominal features
CATEGORICAL_FEATURES = ["OSOURCE", "TCODE", "DOMAIN", "STATE", "PVASTATE",
                        "CLUSTER", "INCOME",
                        "CHILD03", "CHILD07", "CHILD12", "CHILD18", "GENDER",
                        "DATASRCE", "SOLP3", "SOLIH", "WEALTH1", "WEALTH2",
                        "GEOCODE", "LIFESRC", "RFA_2R", "RFA_2A",
                        "RFA_2F", "MDMAUD_R", "MDMAUD_F", "MDMAUD_A",
                        "GEOCODE2"]

# Nominal features needing further cleaning treatment
NOMINAL_FEATURES = ["OSOURCE", "TCODE", "RFA_2", "RFA_3", "RFA_4", "RFA_5",
                    "RFA_6", "RFA_7", "RFA_8", "RFA_9", "RFA_10", "RFA_11",
                    "RFA_12", "RFA_13", "RFA_14", "RFA_15", "RFA_16", "RFA_17",
                    "RFA_18", "RFA_19", "RFA_20", "RFA_21", "RFA_22", "RFA_23",
                    "RFA_24"]

ORDINAL_MAPPING_MDMAUD = [
    {'col': 'MDMAUD_R', 'mapping': {'D': 1, 'I': 2, 'L': 3, 'C': 4}},
    {'col': 'MDMAUD_A', 'mapping': {'L': 1, 'C': 2, 'M': 3, 'T': 4}}]

ORDINAL_MAPPING_RFA = [{'col': c,
                        'mapping': {'A': 1, 'B': 2, 'C': 3, 'D': 4,
                                    'E': 5, 'F': 6, 'G': 7}}
                       for c in ['RFA_2A', 'RFA_3A', 'RFA_4A', 'RFA_5A',
                                 'RFA_6A', 'RFA_7A', 'RFA_8A', 'RFA_9A',
                                 'RFA_10A', 'RFA_11A', 'RFA_12A', 'RFA_13A',
                                 'RFA_14A', 'RFA_15A', 'RFA_16A', 'RFA_17A',
                                 'RFA_18A', 'RFA_19A', 'RFA_20A', 'RFA_21A',
                                 'RFA_22A', 'RFA_23A', 'RFA_24A']]

ORDINAL_MAPPING_SOCIOECON = [{'col': 'DOMAINSocioEconomic',
                              'mapping': {'1': 1, '2': 2, '3': 2, '4': 3}}]

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

PROMO_HISTORY_DATES = ["ADATE_2", "ADATE_3", "ADATE_4", "ADATE_5", "ADATE_6",
                       "ADATE_7", "ADATE_8", "ADATE_9", "ADATE_10",
                       "ADATE_11", "ADATE_12", "ADATE_13", "ADATE_14",
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
NA_CODES = {
    "ODATEDW": ["", ".", " "],
    "OSOURCE": ["", ".", " "],
    "TCODE": ["", ".", " "],
    "STATE": ["", ".", " "],
    "ZIP": ["", ".", " "],
    "MAILCODE": ["", "."],
    "PVASTATE": ["", ".", " "],
    "DOB": ["", ".", " "],
    "NOEXCH": ["", ".", " "],
    "RECINHSE": ["", "."],
    "RECP3": ["", "."],
    "RECPGVG": ["", "."],
    "RECSWEEP": ["", "."],
    "MDMAUD": ["", ".", " "],
    "DOMAIN": ["", ".", " "],
    "CLUSTER": ["", ".", " "],
    "AGE": ["", ".", " "],
    "AGEFLAG": ["", ".", " "],
    "HOMEOWNR": ["", ".", " "],
    "CHILD03": ["", ".", " "],
    "CHILD07": ["", ".", " "],
    "CHILD12": ["", ".", " "],
    "CHILD18": ["", ".", " "],
    "NUMCHLD": ["", ".", " "],
    "INCOME": ["", ".", " "],
    "GENDER": ["", ".", " "],
    "WEALTH1": ["", ".", " "],
    "HIT": ["", ".", " "],
    "MBCRAFT": ["", ".", " "],
    "MBGARDEN": ["", ".", " "],
    "MBBOOKS": ["", ".", " "],
    "MBCOLECT": ["", ".", " "],
    "MAGFAML": ["", ".", " "],
    "MAGFEM": ["", ".", " "],
    "MAGMALE": ["", ".", " "],
    "PUBGARDN": ["", ".", " "],
    "PUBCULIN": ["", ".", " "],
    "PUBHLTH": ["", ".", " "],
    "PUBDOITY": ["", ".", " "],
    "PUBNEWFN": ["", ".", " "],
    "PUBPHOTO": ["", ".", " "],
    "PUBOPP": ["", ".", " "],
    "DATASRCE": ["", ".", " "],
    "MALEMILI": ["", ".", " "],
    "MALEVET": ["", ".", " "],
    "VIETVETS": ["", ".", " "],
    "WWIIVETS": ["", ".", " "],
    "LOCALGOV": ["", ".", " "],
    "STATEGOV": ["", ".", " "],
    "FEDGOV": ["", ".", " "],
    "SOLP3": ["", ".", " "],
    "SOLIH": ["", ".", " "],
    "MAJOR": ["", "."],
    "WEALTH2": ["", ".", " "],
    "GEOCODE": ["", ".", " "],
    "COLLECT1": ["", "."],
    "VETERANS": ["", "."],
    "BIBLE": ["", "."],
    "CATLG": ["", "."],
    "HOMEE": ["", "."],
    "PETS": ["", "."],
    "CDPLAY": ["", "."],
    "STEREO": ["", "."],
    "PCOWNERS": ["", "."],
    "PHOTO": ["", "."],
    "CRAFTS": ["", "."],
    "FISHER": ["", "."],
    "GARDENIN": ["", "."],
    "BOATS": ["", "."],
    "WALKER": ["", "."],
    "KIDSTUFF": ["", "."],
    "CARDS": ["", "."],
    "PLATES": ["", "."],
    "LIFESRC": ["", ".", " "],
    "PEPSTRFL": ["", "."],
    "POP901": ["", ".", " "],
    "POP902": ["", ".", " "],
    "POP903": ["", ".", " "],
    "POP90C1": ["", ".", " "],
    "POP90C2": ["", ".", " "],
    "POP90C3": ["", ".", " "],
    "POP90C4": ["", ".", " "],
    "POP90C5": ["", ".", " "],
    "ETH1": ["", ".", " "],
    "ETH2": ["", ".", " "],
    "ETH3": ["", ".", " "],
    "ETH4": ["", ".", " "],
    "ETH5": ["", ".", " "],
    "ETH6": ["", ".", " "],
    "ETH7": ["", ".", " "],
    "ETH8": ["", ".", " "],
    "ETH9": ["", ".", " "],
    "ETH10": ["", ".", " "],
    "ETH11": ["", ".", " "],
    "ETH12": ["", ".", " "],
    "ETH13": ["", ".", " "],
    "ETH14": ["", ".", " "],
    "ETH15": ["", ".", " "],
    "ETH16": ["", ".", " "],
    "AGE901": ["", ".", " "],
    "AGE902": ["", ".", " "],
    "AGE903": ["", ".", " "],
    "AGE904": ["", ".", " "],
    "AGE905": ["", ".", " "],
    "AGE906": ["", ".", " "],
    "AGE907": ["", ".", " "],
    "CHIL1": ["", ".", " "],
    "CHIL2": ["", ".", " "],
    "CHIL3": ["", ".", " "],
    "AGEC1": ["", ".", " "],
    "AGEC2": ["", ".", " "],
    "AGEC3": ["", ".", " "],
    "AGEC4": ["", ".", " "],
    "AGEC5": ["", ".", " "],
    "AGEC6": ["", ".", " "],
    "AGEC7": ["", ".", " "],
    "CHILC1": ["", ".", " "],
    "CHILC2": ["", ".", " "],
    "CHILC3": ["", ".", " "],
    "CHILC4": ["", ".", " "],
    "CHILC5": ["", ".", " "],
    "HHAGE1": ["", ".", " "],
    "HHAGE2": ["", ".", " "],
    "HHAGE3": ["", ".", " "],
    "HHN1": ["", ".", " "],
    "HHN2": ["", ".", " "],
    "HHN3": ["", ".", " "],
    "HHN4": ["", ".", " "],
    "HHN5": ["", ".", " "],
    "HHN6": ["", ".", " "],
    "MARR1": ["", ".", " "],
    "MARR2": ["", ".", " "],
    "MARR3": ["", ".", " "],
    "MARR4": ["", ".", " "],
    "HHP1": ["", ".", " "],
    "HHP2": ["", ".", " "],
    "DW1": ["", ".", " "],
    "DW2": ["", ".", " "],
    "DW3": ["", ".", " "],
    "DW4": ["", ".", " "],
    "DW5": ["", ".", " "],
    "DW6": ["", ".", " "],
    "DW7": ["", ".", " "],
    "DW8": ["", ".", " "],
    "DW9": ["", ".", " "],
    "HV1": ["", ".", " "],
    "HV2": ["", ".", " "],
    "HV3": ["", ".", " "],
    "HV4": ["", ".", " "],
    "HU1": ["", ".", " "],
    "HU2": ["", ".", " "],
    "HU3": ["", ".", " "],
    "HU4": ["", ".", " "],
    "HU5": ["", ".", " "],
    "HHD1": ["", ".", " "],
    "HHD2": ["", ".", " "],
    "HHD3": ["", ".", " "],
    "HHD4": ["", ".", " "],
    "HHD5": ["", ".", " "],
    "HHD6": ["", ".", " "],
    "HHD7": ["", ".", " "],
    "HHD8": ["", ".", " "],
    "HHD9": ["", ".", " "],
    "HHD10": ["", ".", " "],
    "HHD11": ["", ".", " "],
    "HHD12": ["", ".", " "],
    "ETHC1": ["", ".", " "],
    "ETHC2": ["", ".", " "],
    "ETHC3": ["", ".", " "],
    "ETHC4": ["", ".", " "],
    "ETHC5": ["", ".", " "],
    "ETHC6": ["", ".", " "],
    "HVP1": ["", ".", " "],
    "HVP2": ["", ".", " "],
    "HVP3": ["", ".", " "],
    "HVP4": ["", ".", " "],
    "HVP5": ["", ".", " "],
    "HVP6": ["", ".", " "],
    "HUR1": ["", ".", " "],
    "HUR2": ["", ".", " "],
    "RHP1": ["", ".", " "],
    "RHP2": ["", ".", " "],
    "RHP3": ["", ".", " "],
    "RHP4": ["", ".", " "],
    "HUPA1": ["", ".", " "],
    "HUPA2": ["", ".", " "],
    "HUPA3": ["", ".", " "],
    "HUPA4": ["", ".", " "],
    "HUPA5": ["", ".", " "],
    "HUPA6": ["", ".", " "],
    "HUPA7": ["", ".", " "],
    "RP1": ["", ".", " "],
    "RP2": ["", ".", " "],
    "RP3": ["", ".", " "],
    "RP4": ["", ".", " "],
    "MSA": ["", ".", " "],
    "ADI": ["", ".", " "],
    "DMA": ["", ".", " "],
    "IC1": ["", ".", " "],
    "IC2": ["", ".", " "],
    "IC3": ["", ".", " "],
    "IC4": ["", ".", " "],
    "IC5": ["", ".", " "],
    "IC6": ["", ".", " "],
    "IC7": ["", ".", " "],
    "IC8": ["", ".", " "],
    "IC9": ["", ".", " "],
    "IC10": ["", ".", " "],
    "IC11": ["", ".", " "],
    "IC12": ["", ".", " "],
    "IC13": ["", ".", " "],
    "IC14": ["", ".", " "],
    "IC15": ["", ".", " "],
    "IC16": ["", ".", " "],
    "IC17": ["", ".", " "],
    "IC18": ["", ".", " "],
    "IC19": ["", ".", " "],
    "IC20": ["", ".", " "],
    "IC21": ["", ".", " "],
    "IC22": ["", ".", " "],
    "IC23": ["", ".", " "],
    "HHAS1": ["", ".", " "],
    "HHAS2": ["", ".", " "],
    "HHAS3": ["", ".", " "],
    "HHAS4": ["", ".", " "],
    "MC1": ["", ".", " "],
    "MC2": ["", ".", " "],
    "MC3": ["", ".", " "],
    "TPE1": ["", ".", " "],
    "TPE2": ["", ".", " "],
    "TPE3": ["", ".", " "],
    "TPE4": ["", ".", " "],
    "TPE5": ["", ".", " "],
    "TPE6": ["", ".", " "],
    "TPE7": ["", ".", " "],
    "TPE8": ["", ".", " "],
    "TPE9": ["", ".", " "],
    "PEC1": ["", ".", " "],
    "PEC2": ["", ".", " "],
    "TPE10": ["", ".", " "],
    "TPE11": ["", ".", " "],
    "TPE12": ["", ".", " "],
    "TPE13": ["", ".", " "],
    "LFC1": ["", ".", " "],
    "LFC2": ["", ".", " "],
    "LFC3": ["", ".", " "],
    "LFC4": ["", ".", " "],
    "LFC5": ["", ".", " "],
    "LFC6": ["", ".", " "],
    "LFC7": ["", ".", " "],
    "LFC8": ["", ".", " "],
    "LFC9": ["", ".", " "],
    "LFC10": ["", ".", " "],
    "OCC1": ["", ".", " "],
    "OCC2": ["", ".", " "],
    "OCC3": ["", ".", " "],
    "OCC4": ["", ".", " "],
    "OCC5": ["", ".", " "],
    "OCC6": ["", ".", " "],
    "OCC7": ["", ".", " "],
    "OCC8": ["", ".", " "],
    "OCC9": ["", ".", " "],
    "OCC10": ["", ".", " "],
    "OCC11": ["", ".", " "],
    "OCC12": ["", ".", " "],
    "OCC13": ["", ".", " "],
    "EIC1": ["", ".", " "],
    "EIC2": ["", ".", " "],
    "EIC3": ["", ".", " "],
    "EIC4": ["", ".", " "],
    "EIC5": ["", ".", " "],
    "EIC6": ["", ".", " "],
    "EIC7": ["", ".", " "],
    "EIC8": ["", ".", " "],
    "EIC9": ["", ".", " "],
    "EIC10": ["", ".", " "],
    "EIC11": ["", ".", " "],
    "EIC12": ["", ".", " "],
    "EIC13": ["", ".", " "],
    "EIC14": ["", ".", " "],
    "EIC15": ["", ".", " "],
    "EIC16": ["", ".", " "],
    "OEDC1": ["", ".", " "],
    "OEDC2": ["", ".", " "],
    "OEDC3": ["", ".", " "],
    "OEDC4": ["", ".", " "],
    "OEDC5": ["", ".", " "],
    "OEDC6": ["", ".", " "],
    "OEDC7": ["", ".", " "],
    "EC1": ["", ".", " "],
    "EC2": ["", ".", " "],
    "EC3": ["", ".", " "],
    "EC4": ["", ".", " "],
    "EC5": ["", ".", " "],
    "EC6": ["", ".", " "],
    "EC7": ["", ".", " "],
    "EC8": ["", ".", " "],
    "SEC1": ["", ".", " "],
    "SEC2": ["", ".", " "],
    "SEC3": ["", ".", " "],
    "SEC4": ["", ".", " "],
    "SEC5": ["", ".", " "],
    "AFC1": ["", ".", " "],
    "AFC2": ["", ".", " "],
    "AFC3": ["", ".", " "],
    "AFC4": ["", ".", " "],
    "AFC5": ["", ".", " "],
    "AFC6": ["", ".", " "],
    "VC1": ["", ".", " "],
    "VC2": ["", ".", " "],
    "VC3": ["", ".", " "],
    "VC4": ["", ".", " "],
    "ANC1": ["", ".", " "],
    "ANC2": ["", ".", " "],
    "ANC3": ["", ".", " "],
    "ANC4": ["", ".", " "],
    "ANC5": ["", ".", " "],
    "ANC6": ["", ".", " "],
    "ANC7": ["", ".", " "],
    "ANC8": ["", ".", " "],
    "ANC9": ["", ".", " "],
    "ANC10": ["", ".", " "],
    "ANC11": ["", ".", " "],
    "ANC12": ["", ".", " "],
    "ANC13": ["", ".", " "],
    "ANC14": ["", ".", " "],
    "ANC15": ["", ".", " "],
    "POBC1": ["", ".", " "],
    "POBC2": ["", ".", " "],
    "LSC1": ["", ".", " "],
    "LSC2": ["", ".", " "],
    "LSC3": ["", ".", " "],
    "LSC4": ["", ".", " "],
    "VOC1": ["", ".", " "],
    "VOC2": ["", ".", " "],
    "VOC3": ["", ".", " "],
    "HC1": ["", ".", " "],
    "HC2": ["", ".", " "],
    "HC3": ["", ".", " "],
    "HC4": ["", ".", " "],
    "HC5": ["", ".", " "],
    "HC6": ["", ".", " "],
    "HC7": ["", ".", " "],
    "HC8": ["", ".", " "],
    "HC9": ["", ".", " "],
    "HC10": ["", ".", " "],
    "HC11": ["", ".", " "],
    "HC12": ["", ".", " "],
    "HC13": ["", ".", " "],
    "HC14": ["", ".", " "],
    "HC15": ["", ".", " "],
    "HC16": ["", ".", " "],
    "HC17": ["", ".", " "],
    "HC18": ["", ".", " "],
    "HC19": ["", ".", " "],
    "HC20": ["", ".", " "],
    "HC21": ["", ".", " "],
    "MHUC1": ["", ".", " "],
    "MHUC2": ["", ".", " "],
    "AC1": ["", ".", " "],
    "AC2": ["", ".", " "],
    "ADATE_2": ["", ".", " "],
    "ADATE_3": ["", ".", " "],
    "ADATE_4": ["", ".", " "],
    "ADATE_5": ["", ".", " "],
    "ADATE_6": ["", ".", " "],
    "ADATE_7": ["", ".", " "],
    "ADATE_8": ["", ".", " "],
    "ADATE_9": ["", ".", " "],
    "ADATE_10": ["", ".", " "],
    "ADATE_11": ["", ".", " "],
    "ADATE_12": ["", ".", " "],
    "ADATE_13": ["", ".", " "],
    "ADATE_14": ["", ".", " "],
    "ADATE_15": ["", ".", " "],
    "ADATE_16": ["", ".", " "],
    "ADATE_17": ["", ".", " "],
    "ADATE_18": ["", ".", " "],
    "ADATE_19": ["", ".", " "],
    "ADATE_20": ["", ".", " "],
    "ADATE_21": ["", ".", " "],
    "ADATE_22": ["", ".", " "],
    "ADATE_23": ["", ".", " "],
    "ADATE_24": ["", ".", " "],
    "RFA_2": ["", ".", " "],
    "RFA_3": ["", ".", " "],
    "RFA_4": ["", ".", " "],
    "RFA_5": ["", ".", " "],
    "RFA_6": ["", ".", " "],
    "RFA_7": ["", ".", " "],
    "RFA_8": ["", ".", " "],
    "RFA_9": ["", ".", " "],
    "RFA_10": ["", ".", " "],
    "RFA_11": ["", ".", " "],
    "RFA_12": ["", ".", " "],
    "RFA_13": ["", ".", " "],
    "RFA_14": ["", ".", " "],
    "RFA_15": ["", ".", " "],
    "RFA_16": ["", ".", " "],
    "RFA_17": ["", ".", " "],
    "RFA_18": ["", ".", " "],
    "RFA_19": ["", ".", " "],
    "RFA_20": ["", ".", " "],
    "RFA_21": ["", ".", " "],
    "RFA_22": ["", ".", " "],
    "RFA_23": ["", ".", " "],
    "RFA_24": ["", ".", " "],
    "CARDPROM": ["", ".", " "],
    "MAXADATE": ["", ".", " "],
    "NUMPROM": ["", ".", " "],
    "CARDPM12": ["", ".", " "],
    "NUMPRM12": ["", ".", " "],
    "RDATE_3": ["", ".", " "],
    "RDATE_4": ["", ".", " "],
    "RDATE_5": ["", ".", " "],
    "RDATE_6": ["", ".", " "],
    "RDATE_7": ["", ".", " "],
    "RDATE_8": ["", ".", " "],
    "RDATE_9": ["", ".", " "],
    "RDATE_10": ["", ".", " "],
    "RDATE_11": ["", ".", " "],
    "RDATE_12": ["", ".", " "],
    "RDATE_13": ["", ".", " "],
    "RDATE_14": ["", ".", " "],
    "RDATE_15": ["", ".", " "],
    "RDATE_16": ["", ".", " "],
    "RDATE_17": ["", ".", " "],
    "RDATE_18": ["", ".", " "],
    "RDATE_19": ["", ".", " "],
    "RDATE_20": ["", ".", " "],
    "RDATE_21": ["", ".", " "],
    "RDATE_22": ["", ".", " "],
    "RDATE_23": ["", ".", " "],
    "RDATE_24": ["", ".", " "],
    "RAMNT_3": ["", ".", " "],
    "RAMNT_4": ["", ".", " "],
    "RAMNT_5": ["", ".", " "],
    "RAMNT_6": ["", ".", " "],
    "RAMNT_7": ["", ".", " "],
    "RAMNT_8": ["", ".", " "],
    "RAMNT_9": ["", ".", " "],
    "RAMNT_10": ["", ".", " "],
    "RAMNT_11": ["", ".", " "],
    "RAMNT_12": ["", ".", " "],
    "RAMNT_13": ["", ".", " "],
    "RAMNT_14": ["", ".", " "],
    "RAMNT_15": ["", ".", " "],
    "RAMNT_16": ["", ".", " "],
    "RAMNT_17": ["", ".", " "],
    "RAMNT_18": ["", ".", " "],
    "RAMNT_19": ["", ".", " "],
    "RAMNT_20": ["", ".", " "],
    "RAMNT_21": ["", ".", " "],
    "RAMNT_22": ["", ".", " "],
    "RAMNT_23": ["", ".", " "],
    "RAMNT_24": ["", ".", " "],
    "RAMNTALL": ["", ".", " "],
    "NGIFTALL": ["", ".", " "],
    "CARDGIFT": ["", ".", " "],
    "MINRAMNT": ["", ".", " "],
    "MINRDATE": ["", ".", " "],
    "MAXRAMNT": ["", ".", " "],
    "MAXRDATE": ["", ".", " "],
    "LASTGIFT": ["", ".", " "],
    "LASTDATE": ["", ".", " "],
    "FISTDATE": ["", ".", " "],
    "NEXTDATE": ["", ".", " "],
    "TIMELAG": ["", ".", " "],
    "AVGGIFT": ["", ".", " "],
    "CONTROLN": ["", ".", " "],
    "TARGET_B": ["", ".", " "],
    "TARGET_D": ["", ".", " "],
    "HPHONE_D": ["", ".", " "],
    "RFA_2R": ["", ".", " "],
    "RFA_2F": ["", ".", " "],
    "RFA_2A": ["", ".", " "],
    "MDMAUD_R": ["", ".", " "],
    "MDMAUD_F": ["", ".", " "],
    "MDMAUD_A": ["", ".", " "],
    "CLUSTER2": ["", ".", " "],
    "GEOCODE2": ["", ".", " "]
}


class KDD98DataProvider:
    """ Handles dataset loading and stores datasets in hdf store
    for speedy future access.
    Expects input data as distributed on UCI's machine learning repository
    (either 'cup98LRN.txt' or 'cup98VAL.txt').
    """

    # Several features need special treatment on import.
    # Where necessary, these are included here for explicit datatype casting.
    # The rest of the features will be guessed by pandas on reading the CSV.
    dtype_specs = {}
    for b in BINARY_FEATURES:
        dtype_specs[b] = 'str'
    for c in CATEGORICAL_FEATURES:
        dtype_specs[c] = 'category'
    for n in NOMINAL_FEATURES:
        dtype_specs[n] = 'str'
    for d in DATE_FEATURES:
        dtype_specs[d] = 'str'
    dtype_specs['TARGET_B'] = 'int'

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
        self._raw_data = {}
        self._cleaned_data = {}
        self._numeric_data = {}
        self._imputed_data = {}
        self._ar_data = {}

        self.download_url = download_url
        self.reference_date = Config.get("reference_date")

        if csv_file in Config.get("learn_file_name",
                                  "learn_test_file_name",
                                  "validation_file_name"):
            self.raw_data_name = pathlib.Path(csv_file).stem
            logger.info("Set raw data file name to: {}"
                        .format(self.raw_data_name))
            if "lrn" in csv_file.lower():
                self.cleaned_data_name = Config.get("learn_cleaned_name")
                self.num_data_name = Config.get("learn_numeric_name")
                self.imp_data_name = Config.get("learn_imputed_name")
                self.ar_data_name = Config.get("learn_ar_name")
                self.fit_transformations = True
            elif "val" in csv_file.lower():
                self.cleaned_data_name = Config.get("validation_cleaned_name")
                self.num_data_name = Config.get("validation_numeric_name")
                self.imp_data_name = Config.get("validation_imputed_name")
                self.ar_data_name = Config.get("validation_ar_name")
                self.fit_transformations = False
        else:
            raise ValueError("Set csv_file to either training (kdd98LRN.txt) or validation (kdd98VAL.txt) file.")

    @property
    def raw_data(self):
        if not self._raw_data:
            self.provide("raw")
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value):
        self._raw_data = value

    @property
    def cleaned_data(self):
        if not self._cleaned_data:
            self.provide("cleaned")
        return self._cleaned_data

    @cleaned_data.setter
    def cleaned_data(self, value):
        self._cleaned_data = value

    @property
    def numeric_data(self):
        if not self._numeric_data:
            self.provide("numeric")
        return self._numeric_data

    @numeric_data.setter
    def numeric_data(self, value):
        self._numeric_data = value

    @property
    def imputed_data(self):
        if not self._imputed_data:
            self.provide("imputed")
        return self._imputed_data

    @imputed_data.setter
    def imputed_data(self, value):
        self._imputed_data = value

    @property
    def all_relevant_data(self):
        if not self._ar_data:
            self.provide("all_relevant")
        return self._ar_data

    @all_relevant_data.setter
    def all_relevant_data(self, value):
        self._ar_data = value

    def provide(self, type):
        """
        Provides data by first checking the hdf store, then loading csv data.

        If cleaned data is requested, the returned pandas object has
        - binary
        - numeric (float, int)
        - ordinal / nominal categorical
        - all missing values np.nan
        - dates in np.datetime64

        If numeric data is requested, the returned pandas object has
        - date features transformed to time deltas
        - ZIP transformed to coordinates
        - encoded categoricals

        in it.

        Params
        ------
        type    One of ["raw", "preproc", "numeric", "imputed", "all_relevant"].
                Raw is as read by pandas, cleaned is with
                preprocessing operations applied.
        """
        name_mapper = {
            "raw": {
                "key": self.raw_data_name,
                "data_attrib": "_raw_data"},
            "cleaned": {
                "key": self.cleaned_data_name,
                "data_attrib": "_cleaned_data"},
            "numeric": {
                "key": self.num_data_name,
                "data_attrib": "_numeric_data"},
            "imputed": {
                "key": self.imp_data_name,
                "data_attrib": "_imputed_data"},
            "all_relevant": {
                "key": self.ar_data_name,
                "data_attrib": "_ar_data"
            }
        }

        assert(type in ["raw", "cleaned", "numeric", "imputed", "all_relevant"])

        try:
            # First, try to load the data from hdf
            # and set the object
            data = self._unpickle_df(name_mapper[type]["key"])
            setattr(self, name_mapper[type]["data_attrib"], data)
        except Exception:
            # If it fails and we ask for cleaned data,
            # try to find the raw data in hdf and, if present,
            # load it. If we ask for cleaned data, try to find
            # cleaned data in hdf and load if present.
            if type == "cleaned":
                try:
                    self.provide("raw")
                except Exception as e:
                    logger.error("Failed to provide raw data. "
                                 "Cannot provide cleaned data. Reason: {}"
                                 .format(e))
                try:
                    pre = Preprocessor(self)
                    self.cleaned_data = pre.apply_transformation()
                except Exception as e:
                    logger.error("Failed to preprocess raw data.\nReason: {}"
                                 .format(e))
                    raise e
                self._pickle_df(self.cleaned_data, self.cleaned_data_name)
            elif type == "numeric":
                try:
                    self.provide("cleaned")
                except Exception as e:
                    logger.error("Failed to provide cleaned data.\n"
                                 "Cannot provide numeric data. Reason: {}"
                                 .format(e))
                    raise e
                try:
                    eng = Engineer(self)
                    self.numeric_data = eng.apply_transformation()
                except Exception as e:
                    logger.error("Failed to engineer cleaned data.\n"
                                 "Reason: {}".format(e))
                    raise e
                self._pickle_df(self.numeric_data, self.num_data_name)
            elif type == "imputed":
                try:
                    self.provide("numeric")
                except Exception as e:
                    logger.error("Failed to provide numeric data.\n"
                        "Cannot provide imputed data. Reason: {}"
                        .format(e))
                    raise e
                try:    
                    imp = Imputer(self)
                    self.imputed_data = imp.apply_transformation()
                except Exception as e:
                    logger.error("Failed to impute numeric data.\n"
                                "Reason: {}".format(e))
                    raise e
                self._pickle_df(self.imputed_data, self.imp_data_name)
            elif type == "all_relevant":
                try:
                    self.provide("imputed")
                except Exception as e:
                    logger.error("Failed to provide imputed data.\n"
                        "Cannot provide all-relevant data. Reason: {}"
                        .format(e))
                    raise e
                try:
                    fext = Extractor(self)
                    self.all_relevant_data = fext.apply_transformation()
                except Exception as e:
                    logger.error("Failed to extract features.\n"
                                 "Reason: {}".format(e))
                    raise e
                self._pickle_df(self.all_relevant_data, self.ar_data_name)
            else:
                try:
                    self._read_csv_data()
                except Exception as error:
                    logger.error(
                        "Failed to load data from csv file {}!"
                        .format(self.raw_data_file_name))
                    raise error

    def _read_csv_data(self):
        """
        Read in csv data. After successful read,
        raw data is saved to HDF for future access.
        """

        try:
            data_file = Config.get("data_dir") / self.raw_data_file_name
            if not data_file.is_file():
                logger.info("Data not stored locally. Downloading...")
                try:
                    self._fetch_online(self.download_url)
                except urllib.error.HTTPError:
                    logger.error(
                        "Failed to download dataset from: {}."
                        .format(self.download_url))

            logger.info("Reading csv file: " + self.raw_data_file_name)
            raw_data = pd.read_csv(
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

        if "val" in self.raw_data_file_name.lower():
            try:
                target_file = Config.get("data_dir") / Config.get("validation_target_file_name")
                logger.info("Reading csv file: " + Config.get("validation_target_file_name"))
                targets = pd.read_csv(
                    pathlib.Path(Config.get("data_dir"), Config.get("validation_target_file_name")),
                    index_col=INDEX_NAME,
                    dtype={"TARGET_B": "Int64", "TARGET_D": "float64"},
                    low_memory=False,  # needed for mixed type columns
                    memory_map=True  # load file in memory
                )
                raw_data = raw_data.merge(targets, on=INDEX_NAME)
            except Exception as exc:
                logger.error(exc)
                raise

        self.raw_data = {
            "data": raw_data.drop(TARGETS, axis=1),
            "feature_names": raw_data.drop(TARGETS, axis=1).columns.values.tolist(),
            "target_names": TARGETS,
            "targets": raw_data.loc[:,TARGETS],
            "stage": "raw"
        }


        self._pickle_df(self.raw_data, self.raw_data_name)

    def _unpickle_df(self, key_name):
        """ Loads data from hdf store.
        Raises an error if the key or the file is not found.

        Params
        ------
        key_name    The key to load

        """

        file = key_name + "_pd.pkl"
        try:
            with open(pathlib.Path(Config.get("df_store"), file), "rb") as df:
                dataset = pkl.load(df)
        except(IOError, FileNotFoundError) as error:
            logger.info("No pickled df for '{}' found.".format(key_name))
            raise(error)
        return dataset

    def _pickle_df(self, data, key_name):
        """ Save a pandas dataframe to hdf store. The hdf format 'table' is
        used, which is slower but supports pandas data types. Theoretically,
        it also allows to query the object and return subsets.

        Params
        ------
        data    A pandas dataframe or other object
        key_name    The key name to store the object at.
        """
        file = key_name + "_pd.pkl"
        pathlib.Path(Config.get("df_store")).mkdir(parents=True, exist_ok=True)
        try:
            with open(pathlib.Path(Config.get("df_store"), file), "wb") as df:
                pkl.dump(data, df)
        except Exception as e:
            logger.error(e)
            raise e

    def _fetch_online(self, url=None, dl_dir=None):
        """
        Fetches the data from the specified url
        or from the UCI machine learning database.

        Params:
        url:    Optional url to fetch from.
                Default is UCI machine learning database.
        """

        if not url:
            url = Config.get("download_url")

        if dl_dir:
            path = pathlib.Path(dl_dir)
        else:
            path = pathlib.Path(Config.get("data_dir"))
        path.mkdir(parents=True, exist_ok=True)
        contents = [f for f in path.iterdir()]
        missing = set(Config.get('download_files')) - set(contents)
        print("Files missing: {}".format(missing))
        if missing:
            for f in missing:
                file = path / f
                urllib.request.urlretrieve(url + '/' + f, file)
                if(pathlib.Path(f).suffix == '.zip'):
                    with zipfile.ZipFile(file, mode='r') as archive:
                        archive.extractall(path=path)


class KDD98DataTransformer:

    transformer_config = OrderedDict()

    def __init__(self, data_loader):
        self.dl = data_loader
        self.dataset = None
        self.drop_features = set()
        self.step = "UNDEFINED"
        self.fit = data_loader.fit_transformations

        try:
            pathlib.Path(Config.get("model_store")).mkdir(
                parents=True, exist_ok=True)
            pathlib.Path(Config.get("model_store_internal")).mkdir(
                parents=True, exist_ok=True)
        except Exception as e:
            message = "Failed to create model store directory '{}'.\n"\
                      "Check permissions!\nException message: {}".format(
                          pathlib.Path(Config.get("model_store")), e)
            logger.error(message)
            raise(RuntimeError(message))

    def drop_if_exists(self, dataset, features):
        """
        Drops features if they exist in the dataframe.
        Silently ignores errors if feature is not present.
        """

        for f in features:
            try:
                dataset["data"].drop(f, axis=1, inplace=True)
                logger.info("Dropped feature {} from dataset"
                            .format(f))
            except KeyError:
                logger.info("Tried dropping feature {}, "
                            "but it was not present in the data."
                            "Possibly alreay removed earlier.".format(f))
            except Exception as e:
                logger.info("Removing feature {} failed for reason {}"
                            .format(f, e))
            dataset["feature_names"] = dataset["data"].columns.values.tolist()
        return dataset

    def pre_steps(self):
        return self.dataset

    def post_steps(self):
        return self.dataset

    def process_transformers(self):
        """
        Works on a set of predefined transformers,
        applying each one consecutively to the data.

        It also keeps a list of obsolete features that
        can be removed afterwards.

        A transformer's definition is as follows (example is a date formatter):

        transformer:    A ColumnTransformer object
        dtype:  new datatype for the features transformed.
                If None, will be autoguessed by pandas
        file:   The pickle file where the fitted transformer is located
        drop:   Any features no longer needed after transformation go here

        > "format_dates": {
        >         "transformer": ColumnTransformer([
        >             ("date_format",
        >             DateFormatter(),
        >             DATE_FEATURES)
        >         ]),
        >         "dtype": None,
        >         "file": "date_format_transformer.pkl",
        >         "drop": []
        >     },

        Params
        ------
        dataset:   The dataset to process
        transformer_config: A dict containing an ordered sequence
                            of transformers to apply.
        fit:    Whether to train the transformers
                on the data (learning data set) or only
                apply fitted transformers (test/validation data). Default True
        """
        features = self.dataset["data"]
        if self.fit:
            target = self.dataset["targets"]
        else:
            target = None
        drop_features = set()

        for t, c in self.transformer_config.items():
            logging.info("Working on transformer '{}'".format(t))
            transformed, transformer = [None]*2
            if self.fit:
                transformer = c["transformer"]
                try:
                    transformed = transformer.fit_transform(features, target)
                except Exception as e:
                    message = "Failed to fit_transform with '{}'"\
                              ". Message: {}".format(t, e)
                    logger.error(message)
                    raise RuntimeError(message)
                features = ut.update_df_with_transformed(
                    features, transformed, transformer, c["dtype"])

                with open(pathlib.Path(
                        Config.get("model_store_internal"), c["file"]), "wb") as ms:
                    pkl.dump(transformer, ms)
            else:
                try:
                    with open(pathlib.Path(
                            Config.get("model_store_internal"), c["file"]), "rb") as ms:
                        transformer = pkl.load(ms)
                except Exception:
                    message = "Failed to load fitted transformer {}.\n"\
                              "Process kdd98LRN.txt first to learn transformations."\
                              "Aborting...".format(c["file"])
                    logger.error(message)
                    raise(RuntimeError(message))
                try:
                    transformed = transformer.transform(features)
                except Exception as e:
                    message = "Failed to transform with {}.\n"\
                              "Aborting...".format(t)
                    logger.error(message)
                    raise e
                features = ut.update_df_with_transformed(
                    features, transformed, transformer, c["dtype"])
            drop_features.update(c["drop"])
        self.dataset["data"] = features
        self.dataset["feature_names"] = features.columns.values.tolist()
        return (self.dataset, drop_features)

    def apply_transformation(self, fit=True):
        logger.info("Transformation step {} started...".format(self.step))

        self.pre_steps()

        self.dataset, drop = self.process_transformers()
        self.drop_features.update(drop)

        # Now, drop all features marked for removal
        logger.info("About to drop the following"
                    " features in transformation {}: {}"
                    .format(self.step, sorted(self.drop_features)))
        self.dataset = self.drop_if_exists(self.dataset, self.drop_features)

        self.post_steps()

        logger.info("Transformation step {} completed..."
                    .format(self.step))
        return self.dataset


class Preprocessor(KDD98DataTransformer):

    transformer_config = OrderedDict({
        "dates": {
            "transformer": ColumnTransformer([
                ("date_format",
                 DateFormatter(),
                 DATE_FEATURES)
            ]),
            "dtype": "str",
            "file": "date_format_transformer.pkl",
            "drop": []
        },
        "zipcode": {
            "transformer": ColumnTransformer([
                ("zip_format",
                 ZipFormatter(),
                 ["ZIP"])
            ]),
            "dtype": "int64",
            "file": "zip_format_transformer.pkl",
            "drop": []
        },
        "noexch": {
            "transformer": ColumnTransformer([
                ("noexch_format",
                 NOEXCHFormatter(),
                 ["NOEXCH"])
            ]),
            "dtype": "Int64",
            "file": "noexch_format_transformer.pkl",
            "drop": []
        },
        "mdmaud": {
            "transformer": ColumnTransformer([
                ("format_mdmaud",
                 MDMAUDFormatter(),
                 ["MDMAUD_R", "MDMAUD_F", "MDMAUD_A"])
            ]),
            "dtype": None,
            "file": "mdmaud_format_transformer.pkl",
            "drop": []
        },
        "rfa": {
            "transformer": ColumnTransformer([
                ("fix_rfa",
                 RFAFixer(),
                 NOMINAL_FEATURES[2:])
            ]),
            "dtype": None,
            "file": "fix_rfa.pkl",
            "drop": []
        },
        "ramount": {
            "transformer": ColumnTransformer([
                ("fix_ramount",
                 RAMNTFixer(),
                 GIVING_HISTORY + GIVING_HISTORY_DATES)
            ]),
            "dtype": None,
            "file": "fix_ramount_features.pkl",
            "drop": []
        },
        "binary": {
            "transformer": ColumnTransformer([
                ("binary_x_bl",
                 BinaryFeatureRecode(
                     value_map={"true": "X", "false": " "},
                     correct_noisy=False),
                 ["PEPSTRFL", "MAJOR", "RECINHSE",
                  "RECP3", "RECPGVG", "RECSWEEP"]),
                ("binary_y_blank",
                 BinaryFeatureRecode(
                     value_map={"true": "Y", "false": " "},
                     correct_noisy=False),
                 ["COLLECT1", "VETERANS", "BIBLE", "CATLG",
                  "HOMEE", "PETS", "CDPLAY", "STEREO", "PCOWNERS",
                  "PHOTO", "CRAFTS", "FISHER", "GARDENIN", "BOATS",
                  "WALKER", "KIDSTUFF", "CARDS", "PLATES"]),
                ("binary_e_i",
                 BinaryFeatureRecode(
                     value_map={"true": "E", "false": "I"},
                     correct_noisy=False),
                 ["AGEFLAG"]),
                ("binary_h_u",
                 BinaryFeatureRecode(
                     value_map={"true": "H", "false": "U"},
                     correct_noisy=False),
                 ["HOMEOWNR"]),
                ("binary_b_bl",
                 BinaryFeatureRecode(
                     value_map={"true": "B", "false": " "},
                     correct_noisy=False),
                 ["MAILCODE"]),
                ("binary_1_0",
                 BinaryFeatureRecode(
                     value_map={"true": "1", "false": "0"},
                     correct_noisy=False),
                 ["HPHONE_D", "NOEXCH"])
            ]),
            "dtype": "Int64",
            "file": "binary_transformer.pkl",
            "drop": []
        },
        "multibyte": {
            "transformer": ColumnTransformer([
                ("spread_rfa",
                 MultiByteExtract(["R", "F", "A"]),
                 NOMINAL_FEATURES[3:]),
                ("spread_domain",
                 MultiByteExtract(["Urbanicity", "SocioEconomic"]),
                 ["DOMAIN"])
            ]),
            "dtype": "category",
            "file": "multibyte_transformer.pkl",
            "drop": NOMINAL_FEATURES[2:] + ["DOMAIN"]
        },
        "ordinal": {
            "transformer": ColumnTransformer([
                ("order_mdmaud",
                 OrdinalEncoder(mapping=ORDINAL_MAPPING_MDMAUD,
                                handle_missing="return_nan"),
                 ["MDMAUD_R", "MDMAUD_A"]),
                ("order_rfa",
                 OrdinalEncoder(mapping=ORDINAL_MAPPING_RFA,
                                handle_missing="return_nan",
                                handle_unknown="return_nan"),
                 ["RFA_" + str(i) + "A" for i in range(2, 25)]),
                ("recode_socioecon",
                 OrdinalEncoder(mapping=ORDINAL_MAPPING_SOCIOECON,
                                handle_missing="return_nan",
                                handle_unknown="return_nan"),
                 ["DOMAINSocioEconomic"]),
                ("order_remaining",
                 OrdinalEncoder(handle_missing="return_nan"),
                 ["WEALTH1", "WEALTH2", "INCOME", "MDMAUD_F"] +
                 ["RFA_" + str(i) + "F" for i in range(2, 25)])
            ]),
            "dtype": "Int64",
            "file": "ordinal_transformer.pkl",
            "drop": []
        }
    })

    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.dataset = self.dl.raw_data
        self.dimension_cols = None
        self.step = "Preprocessing"
        self.drop_features = set(DROP_INITIAL + DROP_REDUNDANT)

    def post_steps(self):
        # We train the zero variance dropper, which will populate a list of 
        # features to remove in _dropped.
        
        if self.fit:
            zv = ZeroVarianceSparseDropper(override=['TARGET_B', 'TARGET_D'])
            _ = zv.fit_transform(self.dataset["data"])
            with open(pathlib.Path(
                        Config.get("model_store_internal"), "zero_var_sparse_dropper.pkl"), "wb") as ms:
                    pkl.dump(zv, ms)
        else:
            try:
                with open(pathlib.Path(
                        Config.get("model_store_internal"), "zero_var_sparse_dropper.pkl"), "rb") as ms:
                    zv = pkl.load(ms)
                _ = zv.transform(self.dataset["data"])
            except Exception as e:
                "Failed to load pickled transformer ZeroVarianceSparseDropper. Aborting...."
                raise(e)

        logger.info("About to drop these sparse / constant features: {}"
                    .format(sorted(zv._dropped)))
        self.dataset = self.drop_if_exists(self.dataset, zv._dropped)

        remaining_object_features = self.dataset["data"].select_dtypes(include="object").columns.values.tolist()
        remaining_without_dates = [r for r in remaining_object_features
                                   if r not in DATE_FEATURES]
        if remaining_without_dates:
            logger.warning("After preprocessing, the following (object) features"
                           " were left untreated and automatically"
                           " coerced to 'category' (nominal): {}"
                           .format(remaining_without_dates))
            self.dataset["data"][remaining_without_dates] = self.dataset["data"][remaining_without_dates].astype("category")


class Engineer(KDD98DataTransformer):

    # Since we dynamically determine categorical features, the class
    # variable is overridden by the init function here.
    transformer_config = OrderedDict()

    def filter_features(self, features):
        return [f for f in features if f in self.ALL_FEATURES]

    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.dataset = self.dl.cleaned_data
        features = self.dataset["data"]
        self.step = "Feature Engineering"
        self.ALL_FEATURES = self.dataset["feature_names"]
        self.CATEGORICAL_FEATURES = features.select_dtypes(include="category").columns.values.tolist()
        self.NUMERICAL_FEATURES = [f for f in features.columns.values.tolist()
                                   if f not in self.CATEGORICAL_FEATURES]
        self.BE_CATEGORICALS = ['OSOURCE', 'TCODE', 'STATE', 'CLUSTER']
        self.OHE_CATEGORICALS = [f for f in self.CATEGORICAL_FEATURES if f not in self.BE_CATEGORICALS]
        self.transformer_config = OrderedDict({
            "zip_to_coords": {
                "transformer": ColumnTransformer([
                    ("zip_to_coords",
                    ZipToCoords(),
                    ["ZIP", "STATE"])
                ]),
                "dtype": None,
                "file": "zip_to_coords_transformer.pkl",
                "drop": ["ZIP"]
            },
            "donation_hist": {
                "transformer": ColumnTransformer([
                    ("months_to_donation",
                     MonthsToDonation(reference_date=pd.datetime(1998, 6, 1)),
                     self.filter_features(PROMO_HISTORY_DATES + GIVING_HISTORY_DATES))
                ]),
                "dtype": "Int64",
                "file": "donation_responses_transformer.pkl",
                "drop": PROMO_HISTORY_DATES + GIVING_HISTORY_DATES
            },
            "timedelta": {
                "transformer": ColumnTransformer([
                    ("time_last_donation",
                     DeltaTime(reference_date=pd.datetime(1997, 6, 1),
                               unit="months"),
                     self.filter_features(["LASTDATE", "MINRDATE", "MAXRDATE", "MAXADATE"])),
                    ("membership_years",
                     DeltaTime(reference_date=pd.datetime(1997, 6, 1),
                               unit="years"),
                     self.filter_features(["ODATEDW"]))
                ]),
                "dtype": "Int64",
                "file": "timedelta_transformer.pkl",
                "drop": ["ODATEDW", "LASTDATE", "MINRDATE",
                         "MAXRDATE", "MAXADATE"]
            },
            "binary_encode_categoricals": {
                "transformer": ColumnTransformer([
                    ("be_osource", BinaryEncoder(handle_missing="indicator"), self.filter_features(['OSOURCE'])),
                    ("be_state", BinaryEncoder(handle_missing="indicator"), self.filter_features(['STATE'])),
                    ("be_cluster", BinaryEncoder(handle_missing="indicator"), self.filter_features(['CLUSTER'])),
                    ("be_tcode", BinaryEncoder(handle_missing="indicator"), self.filter_features(['TCODE']))
                ]),
                "dtype": "Int64",
                "file": "binary_encoding_transformer.pkl",
                "drop": self.BE_CATEGORICALS
            },
            "one_hot_encode_categoricals": {
                "transformer": ColumnTransformer([
                    ("oh",
                     OneHotEncoder(use_cat_names=True, handle_missing="indicator"),
                     self.OHE_CATEGORICALS)
                ]),
                "dtype": "Int64",
                "file": "oh_encoding_transformer.pkl",
                "drop": self.OHE_CATEGORICALS
            }
        })


class Imputer(KDD98DataTransformer):

    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.dataset = self.dl.numeric_data
        self.step = "Imputation (Iterative Imputer)"
        self.transformer_config = OrderedDict({
            "impute_numeric": {
                "transformer":  MedianImputer(),
                "dtype": None,
                "file": "median_imputer.pkl",
                "drop": []
            }
        })


class Extractor(KDD98DataTransformer):

    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.dataset = self.dl.imputed_data
        self.step = "Feature Extraction (Boruta all-relevant)"
        self.transformer_config = OrderedDict({
            "boruta_extractor": {
                "transformer": AllRelevantFeatureFilter(),
                "dtype": None,
                "file": "boruta_extractor.pkl",
                "drop": []
            }
        })

    def process_transformers(self):
        features = self.dataset["data"]
        if self.fit:
            target = self.dataset["targets"]
        else:
            target = None
        drop_features = set()

        for t, c in self.transformer_config.items():
            logging.info("Working on transformer '{}'".format(t))
            transformed, transformer = [None]*2
            if self.fit:
                transformer = c["transformer"]
                try:
                    transformer.fit_transform(features, target)
                except Exception as e:
                    message = "Failed to fit_transform with '{}'"\
                              ". Message: {}".format(t, e)
                    logger.error(message)
                    raise RuntimeError(message)
                drop_features.update([f for f in features.columns.values.tolist() if f not in transformer.feature_names])

                with open(pathlib.Path(
                        Config.get("model_store_internal"), c["file"]), "wb") as ms:
                    pkl.dump(transformer, ms)
            else:
                try:
                    with open(pathlib.Path(
                            Config.get("model_store_internal"), c["file"]), "rb") as ms:
                        transformer = pkl.load(ms)
                except Exception:
                    message = "Failed to load fitted transformer {}.\n"\
                              "Process kdd98LRN.txt first to learn transformations."\
                              "Aborting...".format(c["file"])
                    logger.error(message)
                    raise(RuntimeError(message))
                try:
                    drop_features.update([f for f in features.columns.values.tolist() if f not in transformer.feature_names])
                except Exception as e:
                    message = "Failed to transform with {}.\n"\
                              "Aborting...".format(t)
                    logger.error(message)
                    raise e

        self.dataset["data"] = features
        self.dataset["feature_names"] = features.columns.values.tolist()
        return (self.dataset, drop_features)
