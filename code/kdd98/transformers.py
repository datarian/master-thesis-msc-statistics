# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import copy
import datetime
import hashlib
import logging
import sys

import numpy as np
import pandas as pd
from dateutil import relativedelta
from dateutil.rrule import MONTHLY, YEARLY, rrule
from sklearn.base import BaseEstimator, TransformerMixin

from category_encoders import OrdinalEncoder

# Set up the logger
logging.basicConfig(filename=__name__+'.log', level=logging.ERROR)
logger = logging.getLogger(__name__)


__all__ = ['DropSparseLowVar',
           'BinaryFeatureRecode',
           'MultiByteExtract',
           'RecodeUrbanSocioEconomic',
           'DeltaTime',
           'MonthsToDonation']
           #'HashingEncoder',
           #'OneHotEncoder',
           #'OrdinalEncoder']


class DropSparseLowVar(BaseEstimator, TransformerMixin):
    """ Transformer to drop:

    * low variance
    * sparse (abundance of NaN)
    features.

    Parameters:
    -----------

    * var_threshold float
        Defines the threshold for the variance below which columns
        are interpreted as constant.
    * sparse_threshold loat
        Defines the threshold (percentage) of NaN's in a column, anything
        having greater than this percentage NaN's will be discarded.
    """

    def __init__(self, var_threshold=1e-5, sparse_threshold=0.1,
                 keep_anyways=[]):
        """
        Removes features with either a low variance or
        those that contain only very few non-NAN's.

        Parameters:
        -----------

        var_threshold: float. Anything lower than this
                       is considered constant and dropped.
        sparse_threshold: Minimum percentage of non-NaN's needed to keep
                          a feature
        keep_anyways: List of regex patterns for features to keep regardless of
            variance / sparsity.
        """
        self.thresh_var = var_threshold
        self.thresh_sparse = sparse_threshold
        self.feature_names = []
        self.drop_names = []
        self. is_transformed = False
        self.keep_anyways = keep_anyways

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        nrow = X.shape[0]

        keep_names = set()
        for search in self.keep_anyways:
            print(X.filter(regex=search).columns)
            keep_names.update(X.filter(regex=search).columns)

        sparse_names = set(
            [c for c in X if X[c].count() / nrow >= self.thresh_sparse]) - keep_names

        low_var_names = set([c for c in X.select_dtypes(
            include="number") if X[c].var() <= self.thresh_var]) - keep_names

        self.drop_names = list(sparse_names.union(low_var_names))
        print("Constant features: " + str(low_var_names))
        print("Sparse features: " + str(sparse_names))
        print("Keep anyways features: " + str(keep_names))
        print(self.drop_names.sort())
        return self

    def transform(self, X, y=None):
        X_trans = X.copy()
        X_trans = X_trans.drop(columns=self.drop_names)
        self.feature_names = X_trans.columns
        self.is_transformed = True
        return X_trans

    def get_feature_names(self):
        if not isinstance(self.is_transformed, list):
            raise ValueError("Must be transformed first.")
        return self.feature_names


class MultiByteExtract(BaseEstimator, TransformerMixin):
    """
    This is a transformer for multibyte features. Each byte in such
    a multibyte feature is actually a categorical feature.

    The bytes are spread into separate categoricals.

    Params:
    -------
    field_names: A list with the new names for each byte
                    that is to be spread

    impute: False means missing / malformatted entries will be coded NaN
            If a value is passed, fields will be filled with that instead.

    drop_orig: Whether to drop the original multibyte feature or not.
    """

    def __init__(self, field_names, impute=False, drop_orig=True):
        self.field_names = field_names
        # determines how many bytes to extract
        self.sigbytes = len(self.field_names)
        self.impute = impute
        self.drop_orig = drop_orig
        self.feature_names = []
        self.is_transformed = False

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = list(X.columns)
        return self

    def _fill_missing(self):
        if not self.impute:
            return [np.nan]*self.sigbytes
        else:
            return [self.impute]*self.sigbytes

    def _spread(self, feature, index_name):
        """ Fills the byte dataset for each record
        Params:
        -------
        feature: A pandas series
        """
        # Dict to hold the split bytes
        spread_field = {}
        # Iterate over all rows, fill dict
        for row in pd.DataFrame(feature).itertuples(name=None):
            # row[0] is the index, row[1] the content of the cell
            if not row[1] is np.nan:
                if len(row[1]) == self.sigbytes:
                    spread_field[row[0]] = list(row[1])
                else:
                    # The field is invalid
                    spread_field[row[0]] = self._fill_missing()
            else:
                # handle missing values
                spread_field[row[0]] = self._fill_missing()

        # Create the dataframe, orient=index means
        # we interprete the dict's contents as rows (defaults to columns)
        temp_df = pd.DataFrame.from_dict(
            data=spread_field, orient="index")
        temp_df.columns = ["".join([feature.name, f])
                           for f in self.field_names]
        temp_df.index.name = index_name
        # make sure all fields are categorical
        temp_df = temp_df.astype("category")
        return temp_df

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_trans = pd.DataFrame(index=X.index)
        for f in X.columns:
            new_df = self._spread(X[f], X.index.name)
            X_trans = X_trans.merge(new_df, on=X.index.name, copy=False)
        self.feature_names = list(X_trans.columns)
        self.is_transformed = True
        if not self.drop_orig:
            return X.merge(X_trans, on=X.index.name)
        else:
            return X_trans

    def get_feature_names(self):
        if self.is_transformed:
            return self.feature_names

class RecodeUrbanSocioEconomic(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = None

    def fit(self, X, y=None):
        self.feature_names = ["DOMAINUrbanicity", "DOMAINSocioEconomic"]
        return self

    def transform(self, X, y=None):
        urb_dict = {'1': '1', '2': '2', '3': '2', '4': '3'}
        X_trans = pd.DataFrame(X, columns=self.feature_names).astype('category')
        X_trans.loc[X_trans.DOMAINUrbanicity == 'U', 'DOMAINSocioEconomic'] = X_trans.loc[X_trans.DOMAINUrbanicity == 'U', 'DOMAINSocioEconomic'].map(urb_dict)
        X_trans.DOMAINSocioEconomic = X_trans.DOMAINSocioEconomic.cat.remove_unused_categories()
        return X_trans

    def get_feature_names(self):
        if isinstance(self.feature_names, list):
            return self.feature_names

class BinaryFeatureRecode(BaseEstimator, TransformerMixin):
    """
    Recodes one or more boolean feature(s), imputing missing values.
    The feature is recoded to float64, 1.0 = True, 0.0 = False,
    NaN for missing (by default). This can be changed through correct_noisy

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
        self.is_fit = False

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = list(X.columns)
        self.is_fit = True
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # Dict holding the mapping for data -> boolean
        if self.correct_noisy:
            if type(self.correct_noisy) is dict:
                vmap = self.correct_noisy
            else:
                vmap = {'1': 1.0, '0': 1.0, ' ': 0.0}
        else:
            vmap = {'1': 1.0, '0': 0.0, ' ': np.nan}
        vmap[self.value_map.get('true')] = 1.0
        vmap[self.value_map.get('false')] = 0.0
        temp_df = X.copy()
        try:
            for feature in X:
                # Map values in data to True/False.
                # NA values are propagated.
                temp_df[feature] = temp_df[feature].astype('object').map(
                    vmap, na_action='ignore').astype('float64')
        except Exception as exc:
            logger.exception(exc)
            raise
        else:
            return temp_df

    def get_feature_names(self):
        if self.is_fit:
            return self.feature_names


class DeltaTime(BaseEstimator, TransformerMixin):
    """Computes the duration between a date and a reference date in months.

    Parameters:
    -----------

    reference_date: either a single datetimelike or a series of datetimelike
        For series, the same length as the passed dataframe is expected.
    unit: ['months', 'years']
    """
    def __init__(self, reference_date=pd.datetime(1997, 6, 1), unit='months',suffix=True):
        self.reference_date = reference_date
        if suffix:
            self.feature_suffix = "_DELTA_"+unit.upper()
        else:
            self.feature_suffix = ""
        self.unit = unit
        self.suffix = suffix
        self.feature_names = None

    def get_duration(self, date_pair):
        if not pd.isna(date_pair.target) and not pd.isna(date_pair.ref):
            delta = relativedelta.relativedelta(date_pair.ref, date_pair.target)
            if self.unit.lower() == 'months':
                duration = (delta.years * 12) + delta.months
            elif self.unit.lower() == 'years':
                duration = delta.years + 1
        else:
            logger.info("Failed to calculate time delta. Dates: {} and {}.".format(date_pair.target, date_pair.ref))
            duration = np.nan
        return duration

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        # We need to ensure we have datetime objects.
        # The dateparser has to return int64 to work with sklearn, so
        # we need to recast here.
        X_trans = pd.DataFrame().astype('str')

        for f in X.columns:
            X_temp = pd.DataFrame(columns=['target', 'ref'])
            X_temp['target'] = X[f]
            if isinstance(self.reference_date, pd.Series):
                # we have a series of reference dates
                feature_name = f+"_"+str(self.reference_date.name)+self.feature_suffix
            else:
                feature_name = f+self.feature_suffix

            X_temp['ref'] = self.reference_date
            X_trans[feature_name] = X_temp.apply(self.get_duration,axis=1)

        self.feature_names = X_trans.columns

        return X_trans

    def get_feature_names(self):
        return self.feature_names


class MonthsToDonation(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = list()
        self.is_transformed = False

    def fit(self, X, y=None):
        return self

    def calc_diff(self, row):
        ref = row[0]
        target = row[1]
        if not pd.isna(ref) and not pd.isna(target):
            try:
                duration = relativedelta.relativedelta(ref, target).years * 12
                duration += relativedelta.relativedelta(ref, target).months
            except TypeError as err:
                logger.error("Failed to calculate time delta. " +
                                "Dates: {} and {}\nMessage: {}".format(row[0], row[1],err))
                duration = np.nan
        else:
            duration = np.nan
        return duration

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X_trans = pd.DataFrame(index=X.index)
        for i in range(3, 25):
            feat_name = "MONTHS_TO_DONATION_"+str(i)
            mailing = X[["ADATE_"+str(i), "RDATE_"+str(i)]]
            diffs = mailing.agg(self.calc_diff, axis=1)
            X_trans = X_trans.merge(pd.DataFrame(
                diffs, columns=[feat_name]), on=X_trans.index.name)
            self.feature_names.extend([feat_name])
            self.is_transformed = True
        return X_trans.astype("float64")

    def get_feature_names(self):
        if self.is_transformed:
            return self.feature_names
