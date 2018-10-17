# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import datetime
from dateutil import relativedelta
import pandas as pd
import logging
# Set up the logger
logging.basicConfig(filename=__name__+'.log', level=logging.WARNING)
logger = logging.getLogger(__name__)


__all__ = [
    'RemoveConstantFeatures',
    'BooleanFeatureRecode',
    'MultiByteExtract',
    'ZipCodeFormatter',
    'ParseDates',
    'ComputeAge',
    'MonthsToDonation'
]


class RemoveConstantFeatures(BaseEstimator, TransformerMixin):
    """
    This transformer removes constant features, identified
    by a variance smaller than 'threshold' or a length of unique values of 1

    Params:
    -------
    ignore: Whether to ignore the removal. If True, no features are
            removed.

    threshold: The smallest variance not considered as zero.

    numeric_only: Whether to only look at numeric features or at all.
    """

    def __init__(self, ignore=False, threshold=1e-6, numeric_only=True):
        self.ignore = ignore
        self.numeric_only = numeric_only
        self.threshold = threshold
        self.is_fit = False

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = list(X.columns)
        self.is_fit = True
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

    def get_feature_names(self):
        if self.is_fit:
            return self.feature_names


class MultiByteExtract(BaseEstimator, TransformerMixin):
    def __init__(self, field_names, impute=False, drop_orig=True):
        """
        This is a transformer for multibyte features. Each byte in such
        a multibyte feature is actually a categorical feature.

        The bytes are spread into separate categoricals.

        Params:
        -------
        field_names: A list with the new names for each byte that is to be spread

        impute: False means missing / malformatted entries will be coded NaN
                If a value is passed, fields will be filled with that instead.

        drop_orig: Whether to drop the original multibyte feature or not.
        """
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
        temp_df.columns = ["_".join([feature.name, f])
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

    def get_feature_names(self):
        if self.is_fit:
            return self.feature_names


class ZipCodeFormatter(BaseEstimator, TransformerMixin):

    def __init__(self, format=True):
        self.do_format = format

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = list(X.columns)
        return self

    def transform(self, X, y=None):
        zip_series = X.ZIP.copy()
        zip_series = zip_series.str.replace('-', '').astype('category')
        return pd.DataFrame(zip_series)

    def get_feature_names(self):
        if self.is_fit:
            return self.feature_names


class ParseDates(BaseEstimator, TransformerMixin):

    def __init__(self, treat_errors='coerce'):
        self.is_fitted = False
        self.treat_errors = treat_errors
        self.feature_names = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = X.columns
        self.is_fitted = True
        return self

    def transform(self, X, y=None):

        assert isinstance(X, pd.DataFrame)
        X_trans = X.copy().astype("str")

        def fix_format(d):
            # If the date string is only 3 characters long,
            # the format is probably %yy%m
            # If after filling, we have 00 as month, make it 01
            if not d == 'nan':
                if len(d) == 3:
                    d = d[:2]+"0"+d[2]
                    if d[2:] == "00":
                        d = d[:-1] + "1"
            return d

        def fix_century(d):
            if not pd.isnull(d):
                try:
                    if d.year > 1997:
                        d = d.replace(year=d.year-100)
                except:
                    print("Invalid value! "+d)
            return d

        for f in X_trans.columns:
            X_trans[f] = X_trans[f].map(fix_format)
            try:
                X_trans[f] = pd.to_datetime(
                    X_trans[f], format="%y%m", errors='coerce').map(fix_century)
            except Exception as e:
                print(e)
                raise
        return X_trans

    def get_feature_names(self):
        if not self.is_fitted:
            raise ValueError("Needs to be fitted first!")
        return self.feature_names


class ComputeAge(BaseEstimator, TransformerMixin):

    def __init__(self, reference_date=pd.datetime(1997, 6, 1)):
        self.reference_date = reference_date
        self.reference_date = reference_date
        self.feature_names = list()
        self.is_transformed = False

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X_trans = X.copy()

        def get_age(date):
            if not pd.isnull(date):
                age = relativedelta.relativedelta(
                    self.reference_date, date).years
            else:
                age = 0.0
            return age

        for f in X_trans.columns:
            X_trans[f] = X_trans[f].map(get_age)

        self.feature_names = X_trans.columns
        self.is_transformed = True

        return X_trans

    def get_feature_names(self):
        if self.is_transformed:
            return self.feature_names


class MonthsToDonation(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = list()
        self.is_transformed = False

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X_trans = pd.DataFrame(index=X.index)

        for i in range(3, 25):
            select = ["ADATE_"+str(i), "RDATE_"+str(i)]
            feat_name = "MONTHS_TO_DONATION_"+str(i)
            mailing = X[select]

            def calc_diff(row):
                if any([(pd.isnull(v)) for v in row]):
                    d = 0.0
                else:
                    d = relativedelta.relativedelta(row[1], row[0]).months
                    if d < 0.0:
                        d -= 1
                    else:
                        d += 1
                return d

            diffs = mailing.agg(calc_diff, axis=1)
            X_trans = X_trans.merge(pd.DataFrame(
                diffs, columns=[feat_name]), on=X_trans.index.name)
            self.feature_names.extend([feat_name])

        self.is_transformed = True
        return X_trans

    def get_feature_names(self):
        if self.is_transformed:
            return self.feature_names
