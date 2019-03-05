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

import pandas as pd
from sklearn.compose import ColumnTransformer

import kdd98.utils_transformer as ut
from category_encoders import BinaryEncoder, OneHotEncoder
from kdd98.config import Config
from kdd98.transformers import (BinaryFeatureRecode, DateFormatter, DeltaTime,
                                MDMAUDFormatter, MonthsToDonation,
                                MultiByteExtract, NOEXCHFormatter,
                                OrdinalEncoder,
                                ZipFormatter, CategoricalImputer,
                                NumericImputer)

# Set up the logger
logging.basicConfig(filename=__name__ + '.log', level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    'KDD98DataProvider',
    'Cleaner',
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
    "COLLECT1": ["", ".", " "],
    "VETERANS": ["", ".", " "],
    "BIBLE": ["", ".", " "],
    "CATLG": ["", ".", " "],
    "HOMEE": ["", ".", " "],
    "PETS": ["", ".", " "],
    "CDPLAY": ["", ".", " "],
    "STEREO": ["", ".", " "],
    "PCOWNERS": ["", ".", " "],
    "PHOTO": ["", ".", " "],
    "CRAFTS": ["", ".", " "],
    "FISHER": ["", ".", " "],
    "GARDENIN": ["", ".", " "],
    "BOATS": ["", ".", " "],
    "WALKER": ["", ".", " "],
    "KIDSTUFF": ["", ".", " "],
    "CARDS": ["", ".", " "],
    "PLATES": ["", ".", " "],
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
        self._raw_data = pd.DataFrame()
        self._clean_data = pd.DataFrame()
        self._preprocessed_data = pd.DataFrame()
        self._numeric_data = pd.DataFrame()

        self.download_url = download_url
        self.reference_date = Config.get("reference_date")

        if csv_file in Config.get("learn_file_name",
                                  "learn_test_file_name",
                                  "validation_file_name"):
            self.raw_data_name = pathlib.Path(csv_file).stem
            logger.info("Set raw data file name to: {}"
                        .format(self.raw_data_name))
            if "lrn" in csv_file.lower():
                self.clean_data_name = Config.get("learn_clean_name")
                self.preproc_data_name = Config.get("learn_preproc_name")
                self.num_data_name = Config.get("learn_numeric_name")
            elif "val" in csv_file.lower():
                self.clean_data_name = Config.get("validation_clean_name")
                self.preproc_data_name = Config.get("validation_preproc_name")
                self.num_data_name = Config.get("validation_numeric_name")
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

    @property
    def numeric_data(self):
        if self._numeric_data.empty:
            self.provide("numeric")
        return self._numeric_data

    @numeric_data.setter
    def numeric_data(self, value):
        self._numeric_data = value

    def provide(self, type):
        """
        Provides data by first checking the hdf store, then loading csv data.

        If clean data is requested, the returned pandas object has:
        - binary
        - numeric (float, int)
        - ordinal / nominal categorical
        - all missing values np.nan
        - dates in np.datetime64

        If preprocessed data is requested, the returned pandas object has
        - the contents of cleaned data
        - date features transformed to time deltas

        If numeric data is requested, the returned pandas object has
        - encoded categoricals
        - imputed features

        in it.

        Params
        ------
        type    One of ["raw", "clean", "preproc", "numeric"].
                Raw is as read by pandas, clean is with
                cleaning operations applied.
        """
        name_mapper = {
            "raw": {"key": self.raw_data_name,
                    "data_attrib": "_raw_data"},
            "clean": {"key": self.clean_data_name,
                      "data_attrib": "_clean_data"},
            "preproc": {"key": self.preproc_data_name,
                        "data_attrib": "_preprocessed_data"},
            "numeric": {"key": self.num_data_name,
                        "data_attrib": "_numeric_data"}
        }

        assert(type in ["raw", "clean", "preproc", "numeric"])

        try:
            # First, try to load the data from hdf
            # and set the object
            data = self._unpickle_df(name_mapper[type]["key"])
            setattr(self, name_mapper[type]["data_attrib"], data)
        except Exception:
            # If it fails and we ask for clean data,
            # try to find the raw data in hdf and, if present,
            # load it. If we ask for preprocessed data, try to find
            # cleaned data in hdf and load if present.
            if type == "clean":
                try:
                    self.provide("raw")
                except Exception as e:
                    logger.error("Failed to provide raw data. "
                                 "Cannot provide clean data. Reason: {}"
                                 .format(e))
                try:
                    cln = Cleaner(self)
                    self.clean_data = cln.apply_transformation()
                except Exception as e:
                    logger.error("Failed to clean raw data.\nReason: {}"
                                 .format(e))
                    raise e
                self._pickle_df(self.clean_data, self.clean_data_name)
            elif type == "preproc":
                try:
                    self.provide("clean")
                except Exception as e:
                    logger.error("Failed to provide clean data."
                                 " Cannot provide preprocessed data.\n"
                                 "Reason: {}".format(e))
                    raise e
                try:
                    pre = Preprocessor(self)
                    self.preprocessed_data = pre.apply_transformation()
                except Exception as e:
                    logger.error("Failed to preprocess clean data.\n"
                                 "Reason: {}".format(e))
                    raise e
                self._pickle_df(self.preprocessed_data, self.preproc_data_name)
            elif type == "numeric":
                try:
                    self.provide("preproc")
                except Exception as e:
                    logger.error("Failed to provide preprocessed data.\n"
                                 "Cannot provide numeric data. Reason: {}"
                                 .format(e))
                    raise e
                try:
                    eng = Engineer(self)
                    self.numeric_data = eng.apply_transformation()
                except Exception as e:
                    logger.error("Failed to engineer preprocessed data.\n"
                                 "Reason: {}".format(e))
                    raise e
                self._pickle_df(self.numeric_data, self.num_data_name)
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
        self.data = None
        self.drop_features = set()
        self.step = "UNDEFINED"

        try:
            pathlib.Path(Config.get("model_store")).mkdir(
                parents=True, exist_ok=True)
        except Exception as e:
            message = "Failed to create model store directory '{}'.\n"\
                      "Check permissions!\nException message: {}".format(
                          pathlib.Path(Config.get("model_store")), e)
            logger.error(message)
            raise(RuntimeError(message))

    def drop_if_exists(self, data, features):
        """
        Drops features if they exist in the dataframe.
        Silently ignores errors if feature is not present.
        """

        for f in features:
            try:
                data.drop(f, axis=1, inplace=True)
                logger.info("Dropped feature {} from dataset"
                            .format(f))
            except KeyError:
                logger.info("Tried dropping feature {}, "
                            "but it was not present in the data."
                            "Possibly alreay removed earlier.".format(f))
            except Exception as e:
                logger.info("Removing feature {} failed for reason {}"
                            .format(f, e))
        return data

    def pre_steps(self):
        return self.data

    def post_steps(self):
        return self.data

    def process_transformers(self, fit):
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
        data:   The dataset to process
        transformer_config: A dict containing an ordered sequence
                            of transformers to apply. See corresponding
                            methods Cleaner.clean() and Cleaner.preprocess()
        fit:    Whether to train the transformers
                on the data (learning data set) or only
                apply fitted transformers (test/validation data). Default True
        """
        data = self.data.copy(deep=True)
        drop_features = set()

        for t, c in self.transformer_config.items():
            logging.info("Working on transformer '{}'".format(t))
            transformed, transformer = None, None
            if fit:
                transformer = c["transformer"]
                try:
                    transformed = transformer.fit_transform(data)
                except Exception as e:
                    message = "Failed to fit_transform with '{}'"\
                              ". Message: {}".format(t, e)
                    logger.error(message)
                    raise RuntimeError(message)
                data = ut.update_df_with_transformed(
                    data, transformed, transformer, c["dtype"])

                with open(pathlib.Path(
                        Config.get("model_store"), c["file"]), "wb") as ms:
                    pkl.dump(transformer, ms)
            else:
                try:
                    with open(pathlib.Path(
                            Config.get("model_store", c["file"]), "rb")) as ms:
                        transformer = pkl.load(ms)
                except Exception:
                    message = "Failed to load fitted transformer {}.\n"\
                              "Call function with fit=True first"\
                              "to learn the transformers.\n"\
                              "Aborting preprocessing...".format(t)
                    logger.error(message)
                    raise(RuntimeError(message))
                try:
                    transformed = transformer.transform(data)
                except Exception as e:
                    message = "Failed to transform with {}.\n"\
                              "Aborting preprocessing...".format(t)
                    logger.error(message)
                    raise e
                data = ut.update_df_with_transformed(
                    data, transformed, transformer)
            drop_features.update(c["drop"])
        return (data, drop_features)

    def apply_transformation(self, fit=True):
        logger.info("Transformation step {} started...".format(self.step))

        self.pre_steps()

        self.data, drop = self.process_transformers(fit)
        self.drop_features.update(drop)

        # Now, drop all features marked for removal
        logger.info("About to drop the following"
                    " features in transformation {}: {}"
                    .format(self.step, self.drop_features))
        self.data = self.drop_if_exists(self.data, self.drop_features)

        self.post_steps()

        logger.info("Transformation step {} completed..."
                    .format(self.step))
        return self.data.copy(deep=True)


class Cleaner(KDD98DataTransformer):

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
            "dtype": "Int64",
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
        "binary": {
            "transformer": ColumnTransformer([
                ("binary_x_bl",
                 BinaryFeatureRecode(
                     value_map={"true": "X", "false": " "},
                     correct_noisy=False),
                 ["PEPSTRFL", "MAJOR", "RECINHSE",
                  "RECP3", "RECPGVG", "RECSWEEP"]),
                ("binary_y_n",
                 BinaryFeatureRecode(
                     value_map={"true": "Y", "false": "N"},
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
                 ["HPHONE_D", "NOEXCH", "TARGET_B"])
            ]),
            "dtype": "Int64",
            "file": "binary_transformer.pkl",
            "drop": []
        },
        "multibyte": {
            "transformer": ColumnTransformer([
                ("spread_rfa",
                 MultiByteExtract(["R", "F", "A"]),
                 NOMINAL_FEATURES[2:]),
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
                                handle_missing="return_nan"),
                 ["RFA_" + str(i) + "A" for i in range(2, 25)]),
                ("recode_socioecon",
                 OrdinalEncoder(mapping=ORDINAL_MAPPING_SOCIOECON,
                                handle_missing="return_nan"),
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
        self.data = self.dl.raw_data
        self.dimension_cols = None
        self.step = "Cleaning"
        self.drop_features = set(DROP_INITIAL + DROP_REDUNDANT)

    def post_steps(self):
        remaining_object_features = self.data.select_dtypes(include="object").columns.values.tolist()
        remaining_without_dates = [r for r in remaining_object_features
                                   if r not in DATE_FEATURES]
        if remaining_without_dates:
            logger.warning("After cleaning, the following features"
                           " were left untreated and automatically"
                           " coerced to 'category' (nominal): {}"
                           .format(remaining_without_dates))
            self.data[remaining_without_dates] = self.data[remaining_without_dates].astype("category")


class Preprocessor(KDD98DataTransformer):

    LOW_VAR_SPARSE = {'RAMNT_24', 'PETS', 'STEREO', 'CARDS', 'CDPLAY',
                      'MDMAUD_R', 'SOLIH', 'RAMNT_20', 'CHILD18', 'RAMNT_11',
                      'GARDENIN', 'RAMNT_15', 'RAMNT_17', 'PCOWNERS',
                      'RAMNT_13', 'FISHER', 'RAMNT_21', 'HOMEE', 'BIBLE',
                      'PHOTO', 'RAMNT_19', 'RAMNT_7', 'BOATS', 'CHILD03',
                      'PVASTATE', 'SOLP3', 'CATLG', 'CRAFTS', 'GEOCODE',
                      'RAMNT_9', 'RAMNT_4', 'PLATES', 'VETERANS', 'KIDSTUFF',
                      'RFA_2R', 'RAMNT_3', 'RAMNT_6', 'CHILD12', 'NOEXCH',
                      'COLLECT1', 'RAMNT_23', 'CHILD07', 'RAMNT_5', 'NUMCHLD',
                      'RAMNT_10', 'MDMAUD_A', 'WALKER'}

    def filter_features(self, features):
        return list(set(features) - self.LOW_VAR_SPARSE)

    transformer_config = OrderedDict()

    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.data = self.dl.clean_data
        self.step = "Preprocessing"
        self.transformer_config = OrderedDict({
            "donation_hist": {
                "transformer": ColumnTransformer([
                    ("months_to_donation",
                     MonthsToDonation(),
                     self.filter_features(PROMO_HISTORY_DATES +
                                          GIVING_HISTORY_DATES))
                ]),
                "dtype": None,
                "file": "donation_responses_transformer.pkl",
                "drop": PROMO_HISTORY_DATES + GIVING_HISTORY_DATES
            },
            "timedelta": {
                "transformer": ColumnTransformer([
                    ("time_last_donation",
                     DeltaTime(unit="months"),
                     self.filter_features(["LASTDATE", "MINRDATE",
                                           "MAXRDATE", "MAXADATE"])),
                    ("membership_years",
                     DeltaTime(unit="years"),
                     self.filter_features(["ODATEDW"]))
                ]),
                "dtype": "Int64",
                "file": "timedelta_transformer.pkl",
                "drop": ["ODATEDW", "LASTDATE", "MINRDATE",
                         "MAXRDATE", "MAXADATE"]
            }
        })

    def pre_steps(self):
        logger.info("About to drop these sparse / constant features: {}"
                    .format(self.LOW_VAR_SPARSE))
        self.data = self.drop_if_exists(self.data, self.LOW_VAR_SPARSE)


class Engineer(KDD98DataTransformer):

    # Since we dynamically determine categorical features, the class
    # variable is overridden by the init function here.
    transformer_config = OrderedDict()

    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.data = self.dl.preprocessed_data
        self.step = "Feature Engineering"
        self.CATEGORICAL_FEATURES = self.data.select_dtypes(include="category").columns.values.tolist()
        self.NUMERICAL_FEATURES = [f for f in self.data.columns.values.tolist() if f not in self.CATEGORICAL_FEATURES]
        self.BE_CATEGORICALS = ['OSOURCE', 'TCODE', 'ZIP', 'STATE', 'CLUSTER']
        self.OHE_CATEGORICALS = [f for f in self.CATEGORICAL_FEATURES if f not in self.BE_CATEGORICALS]
        self.transformer_config = OrderedDict({
            # Before dealing with categorical features, we need to impute.
            "impute_categories": {
                "transformer": ColumnTransformer([
                    ("impute_categories",
                     CategoricalImputer(),
                     self.CATEGORICAL_FEATURES)
                ]),
                "dtype": None,
                "file": "impute_cats_transformer.pkl",
                "drop": []
            },
            "binary_encode_categoricals": {
                "transformer": ColumnTransformer([
                    ("be_osource", BinaryEncoder(), ['OSOURCE']),
                    ("be_state", BinaryEncoder(), ['STATE']),
                    ("be_cluster", BinaryEncoder(), ['CLUSTER']),
                    ("be_tcode", BinaryEncoder(), ['TCODE']),
                    ("be_zip", BinaryEncoder(), ['ZIP'])
                ]),
                "dtype": "int64",
                "file": "binary_encoding_transformer.pkl",
                "drop": self.BE_CATEGORICALS
            },
            "one_hot_encode_categoricals": {
                "transformer": ColumnTransformer([
                    ("oh",
                     OneHotEncoder(use_cat_names=True,
                                   handle_unknown="error"),
                     self.OHE_CATEGORICALS)
                ]),
                "dtype": "int64",
                "file": "oh_encoding_transformer.pkl",
                "drop": self.OHE_CATEGORICALS
            },
            "impute_remaining": {
                "transformer": ColumnTransformer([
                    ("impute_numeric",
                     NumericImputer(n_iter=10, initial_strategy="median",
                                    random_state=Config.get("random_seed"),
                                    verbose=1),
                     self.NUMERICAL_FEATURES)
                ]),
                "dtype": None,
                "file": "iterative_impute_numerics.pkl",
                "drop": []
            }
        })
