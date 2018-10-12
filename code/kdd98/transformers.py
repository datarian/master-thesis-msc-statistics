# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging
# Set up the logger
logging.basicConfig(filename=__name__+'.log', level=logging.WARNING)
logger = logging.getLogger(__name__)


__all__ = [
    'RemoveConstantFeatures',
    # 'RemoveSparseFeatures',
    'BooleanFeatureRecode',
    'ZipCodeFormatter'
]


class RemoveConstantFeatures(BaseEstimator, TransformerMixin):
    """
    This transformer removes constant features, identified
    by a variance of zero or a length of unique values of 1
    """

    def __init__(self, ignore=False, threshold=1e-6, numeric_only=True):
        self.ignore = ignore
        self.numeric_only = numeric_only
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.ignore:
            return X
        else:
            for f in X.columns:
                if X[f].var(numeric_only=self.numeric_only) <= self.threshold:
                    X.drop(f, axis=1, inplace=True)
        return X


class BooleanFeatureRecode(BaseEstimator, TransformerMixin):
    """
    Recodes one or more boolean feature(s), imputing missing values.
    The feature is recoded to integer 0/1 for False/True

    Params:
    -------
    correct_noisy: boolean or dict.
                    dict: custom map of nan -> bool mapping
                    True: maps {'1': True, '0': False, ' ': False} + value_map
                    False: maps {'1': True, '0': False, ' ': np.nan} + value_map
    value_map: dict describing which values are mapped to True/False
    """

    def __init__(self, correct_noisy=True, value_map=None):
        self.correct_noisy = correct_noisy
        self.value_map = value_map

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # Dict holding the mapping for data -> boolean
        if self.correct_noisy:
            if type(self.correct_noisy) is dict:
                vmap = self.correct_noisy
            else:
                vmap = {'1': True, '0': False, ' ': False}
        else:
            vmap = {'1': True, '0': False, ' ': np.nan}
        vmap[self.value_map.get('true')] = True
        vmap[self.value_map.get('false')] = False
        temp_df = X.copy()
        try:
            for feature in X:
                # Map values in data to True/False.
                # NA values are propagated.
                temp_df[feature] = temp_df[feature].map(
                    vmap, na_action='ignore').astype('bool')
        except Exception as exc:
            logger.exception(exc)
            raise
        else:
            return temp_df


class ZipCodeFormatter(BaseEstimator, TransformerMixin):

    def __init__(self, format=True):
        self.do_format = format

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        zip_series = X.ZIP.copy()
        zip_series = zip_series.str.replace('-', '').astype('category')
        return pd.DataFrame(zip_series)
