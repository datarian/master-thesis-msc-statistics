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

import kdd98.utils_transformer as ut
import numpy as np
import pandas as pd
from kdd98.config import App
from kdd98.data_loader import KDD98DataLoader
from kdd98.transformers import (BinaryFeatureRecode, DeltaTime,
                                MonthsToDonation, MultiByteExtract,
                                OrdinalEncoder, RecodeUrbanSocioEconomic)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from category_encoders import HashingEncoder, OneHotEncoder

# Set up the logger
logging.basicConfig(filename=__name__+'.log', level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'Cleaner'
]


class Cleaner:

    def __init__(self, data_loader=KDD98DataLoader()):
        self.dl = data_loader
        assert(self.dl.raw_data_file_name in [App.config(
            'learn_file_name'), App.config('validation_file_name')])
        self.dataset = self.dl.get_dataset()

    def clean(self):

        # Binary features
        binary_transformers = ColumnTransformer([
            ("binary_x_bl",
             BinaryFeatureRecode(
                 value_map={'true': 'X', 'false': ' '}, correct_noisy=False),
             ['PEPSTRFL', 'NOEXCH', 'MAJOR', 'RECINHSE',
                 'RECP3', 'RECPGVG', 'RECSWEEP']
             ),
            ("binary_y_n",
             BinaryFeatureRecode(
                 value_map={'true': 'Y', 'false': 'N'}, correct_noisy=False),
             ['COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 'PETS', 'CDPLAY', 'STEREO',
              'PCOWNERS', 'PHOTO', 'CRAFTS', 'FISHER', 'GARDENIN',  'BOATS', 'WALKER', 'KIDSTUFF',
              'CARDS', 'PLATES']
             ),
            ("binary_e_i",
             BinaryFeatureRecode(
                 value_map={'true': "E", 'false': 'I'}, correct_noisy=False),
             ['AGEFLAG']
             ),
            ("binary_h_u",
             BinaryFeatureRecode(
                 value_map={'true': "H", 'false': 'U'}, correct_noisy=False),
             ['HOMEOWNR']),
            ("binary_b_bl",
             BinaryFeatureRecode(
                 value_map={'true': 'B', 'false': ' '}, correct_noisy=False),
             ['MAILCODE']
             ),
            ("binary_1_0",
             BinaryFeatureRecode(
                 value_map={'true': '1', 'false': '0'}, correct_noisy=False),
             ['HPHONE_D']
             )
        ])

        # Categorical Features

        # Ordinals
        multibyte_transformer = ColumnTransformer([
            ("spread",
             MultiByteExtract(["R", "F", "A"]),
             self.dl.nominal_features[2:])
        ])

        domain_transformer = ColumnTransformer([
            ("spread_domain",
             MultiByteExtract(["Urbanicity", "SocioEconomic"]),
             ["DOMAIN"])
        ])

        # Remaining ordinals
        ordinal_transformer = ColumnTransformer([
            ("order_ordinals",
             OrdinalEncoder(mapping=self.dl.ordinal_mapping_mdmaud,
                            handle_unknown='ignore'),
             ['MDMAUD_R', 'MDMAUD_A']),
            ("order_multibytes",
             OrdinalEncoder(mapping=self.dl.ordinal_mapping_rfa,
                            handle_unknown='ignore'),
             list(self.dataset.filter(like="RFA_", axis=1).columns))
        ])

        # Transforming the data (possibly fitting first) and rebuilding the pandas dataframe

        # Fix formatting for ZIP feature
        self.dataset.ZIP = self.dataset.ZIP.str.replace(
            '-', '').replace([' ', '.'], np.nan).astype('int64')
        # Fix binary encoding inconsistency for NOEXCH
        self.dataset.NOEXCH = self.dataset.NOEXCH.str.replace("X", "1")
        # Fix some NA value problems:
        self.dataset[['MDMAUD_R', 'MDMAUD_F', 'MDMAUD_A']] = self.dataset.loc[:, [
            'MDMAUD_R', 'MDMAUD_F', 'MDMAUD_A']].replace('X', np.nan)

        binarys = binary_transformers.fit_transform(self.dataset)
        self.dataset = ut.update_df_with_transformed(
            self.dataset, binarys, binary_transformers)
        multibytes = multibyte_transformer.fit_transform(self.dataset)
        self.dataset = ut.update_df_with_transformed(
            self.dataset, multibytes, multibyte_transformer, drop=self.dl.nominal_features, new_dtype="category")
        domains = domain_transformer.fit_transform(self.dataset)
        self.dataset = ut.update_df_with_transformed(
            self.dataset, domains, domain_transformer)
        ordinals = ordinal_transformer.fit_transform(self.dataset)
        self.dataset = ut.update_df_with_transformed(
            self.dataset, ordinals, ordinal_transformer)

        return self.dataset
