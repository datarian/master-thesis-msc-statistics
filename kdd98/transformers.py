# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import logging
import pathlib
import pickle

import numpy as np
import pandas as pd
from dateutil import relativedelta
from dateutil.rrule import MONTHLY, YEARLY, rrule
from fancyimpute import KNN, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from category_encoders import HashingEncoder, OrdinalEncoder
from geopy.exc import GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Here
from kdd98.config import Config

# Set up the logger
logging.basicConfig(filename=__name__ + '.log', level=logging.ERROR)
logger = logging.getLogger(__name__)


__all__ = ['BinaryFeatureRecode',
           'MultiByteExtract',
           'RecodeUrbanSocioEconomic',
           'DateFormatter',
           'ZipFormatter',
           'NOEXCHFormatter',
           'MDMAUDFormatter',
           'DeltaTime',
           'MonthsToDonation',
           'Hasher',
           'CategoricalImputer',
           'NumericImputer',
           "TargetImputer",
           "RAMNTFixer",
           "ZeroVarianceSparseDropper"]


class NamedFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = None

    def get_feature_names(self):
        if isinstance(self.feature_names, list):
            return self.feature_names
        else:
            raise(ValueError("Transformer {} has to be transformed first, cannot return feature names.".format(
                self.__class__.__name__)))


class MultiByteExtract(NamedFeatureTransformer):
    """
    This is a transformer for multibyte features. Each byte in such
    a multibyte feature is actually a categorical feature.

    The bytes are spread into separate categoricals.

    Params:
    -------
    new_features: A list with the new names for each byte
                    that is to be spread
    """

    def __init__(self, new_features):
        super().__init__()
        self.new_features = new_features
        # determines how many bytes to extract
        self.sigbytes = len(self.new_features)

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = X.columns.values.tolist()
        return self

    def _fill_missing(self):
        return [np.nan] * self.sigbytes

    def _spread(self, feature, index_name):
        """ Fills the byte dataset for each record
        Params:
        -------
        feature: A pandas series
        """
        # Dict to hold the split bytes
        spread_field = {}
        # Iterate over all rows, fill dict
        try:
            for row in pd.DataFrame(feature).itertuples(name=None):
                # row[0] is the index, row[1] the content of the cell
                if isinstance(row[1], str):
                    if len(row[1]) == self.sigbytes:
                        spread_field[row[0]] = list(row[1])
                    else:
                        # The field is invalid
                        spread_field[row[0]] = self._fill_missing()
                else:
                    # handle missing values
                    spread_field[row[0]] = self._fill_missing()
        except Exception as e:
            logger.error(
                "Failed to spread feature '{}' for reason {}".format(feature, e))
            raise e

        # Create the dataframe, orient=index means
        # we interprete the dict's contents as rows (defaults to columns)
        temp_df = pd.DataFrame.from_dict(
            data=spread_field,
            orient="index",
            columns=["".join([feature.name, f]) for f in self.new_features])
        temp_df.index.name = index_name
        # make sure all fields are categorical
        temp_df = temp_df.astype("category")
        return temp_df

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_trans = pd.DataFrame(index=X.index)
        try:
            for f in X.columns:
                new_df = self._spread(X[f], X.index.name)
                X_trans = X_trans.merge(new_df, on=X.index.name)
            self.feature_names = list(X_trans.columns)
        except Exception as e:
            raise e
        return X_trans


class RecodeUrbanSocioEconomic(NamedFeatureTransformer):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.feature_names = X.columns.values.tolist()
        return self

    def transform(self, X, y=None):
        urb_dict = {'1': '1', '2': '2', '3': '2', '4': '3'}
        X_trans = pd.DataFrame(
            X, columns=self.feature_names).astype('category')
        X_trans.loc[X_trans.DOMAINUrbanicity == 'U',
                    'DOMAINSocioEconomic'] = X_trans.loc[X_trans.DOMAINUrbanicity == 'U', 'DOMAINSocioEconomic'].map(urb_dict)
        X_trans.DOMAINSocioEconomic = X_trans.DOMAINSocioEconomic.cat.remove_unused_categories()
        return X_trans


class BinaryFeatureRecode(NamedFeatureTransformer):
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
        super().__init__()
        self.correct_noisy = correct_noisy
        self.value_map = value_map
        self.is_fit = False

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = X.columns.values.tolist()
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # Dict holding the mapping for data -> boolean
        if self.correct_noisy:
            if type(self.correct_noisy) is dict:
                vmap = self.correct_noisy
            else:
                vmap = {'1': 1.0, '0': 1.0, '': 0.0}
        else:
            vmap = {'1': 1.0, '0': 0.0, '': np.nan}
        vmap[self.value_map.get('true')] = 1.0
        vmap[self.value_map.get('false')] = 0.0
        temp_df = X.copy()
        try:
            for feature in X:
                # Map values in data to True/False.
                # NA values are propagated.
                temp_df[feature] = temp_df[feature].astype('object').map(
                    vmap, na_action='ignore').astype('Int64')
        except Exception as exc:
            logger.exception(exc)
            raise
        else:
            return temp_df


class DateFormatter(NamedFeatureTransformer):
    """
    Fixes input errors for date features
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.feature_names = X.columns.values.tolist()
        return self

    def _fix_format(self, d):
        if not pd.isna(d) and not str.lower(d) == "nan":
            if len(d) == 3:
                d = "0" + d
        return d

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X = X.applymap(self._fix_format)
        return X


class ZipFormatter(NamedFeatureTransformer):
    """
    Fixes input errors for zip codes
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.feature_names = X.columns.values.tolist()
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        for f in self.feature_names:
            X[f] = X[f].str.replace(
                "-", "").replace([" ", "."], np.nan).astype("int64")
        return X


class RAMNTFixer(NamedFeatureTransformer):
    """ Fixes RAMNT_ features by checking if there is a value for the
        corresponding RDATE_.
        If there is, the amount is really missing.
        If no date is recorded for receiving a donation, this is strong
        evidence that the example actually has not donated and we can set
        the amount to zero.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        self.feature_names = X.filter(regex="RAMNT_*").columns.values.tolist()
        X_trans = pd.DataFrame(index=X.index)

        def really_missing(example):
            ramnt = None
            if pd.isna(example[0]):
                ramnt = 0 if pd.isna(example[1]) else np.nan
            else:
                ramnt = example[0]
            return ramnt

        for i in range(3, 25):
            X_temp = X[["RAMNT_" + str(i), "RDATE_" + str(i)]]
            X_trans["RAMNT_" + str(i)] = X_temp.agg(really_missing, axis=1)

        return X_trans


class RFAFixer(NamedFeatureTransformer):
    """ Sets invalid RFA features to NaN.
        This occurs if strings are not of length 3.
    """
    def __init__(self):
        super().__init__()

    def validate_value(self, v):
        if not pd.isna(v):
            if not len(v) == 3:
                v = np.nan
        return v

    def fit(self, X, y=None):
        self.feature_names = X.columns.values.tolist()
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        X_trans = X.applymap(self.validate_value)
        return X_trans


class NOEXCHFormatter(NamedFeatureTransformer):
    """ Fixes erroneously encoded binary codes
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.feature_names = X.columns.values.tolist()
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        for f in self.feature_names:
            X[f] = X[f].str.replace("X", "1")
        return X


class MDMAUDFormatter(NamedFeatureTransformer):
    """
    Fixes input errors for MDMAUD features
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.feature_names = X.columns.values.tolist()
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))

        X = X.replace("X", np.nan)
        return X


class DateHandler:

    def __init__(self, reference_date):
        self.ref_date = reference_date
        self.ref_year = self.ref_date.year

    def fix_century(self, d):
        if not pd.isna(d):
            try:
                if d.year > self.ref_year:
                    d = d.replace(year=(d.year - 100))
            except Exception as err:
                logger.warning(
                    "Failed to fix century for date {}, reason: {}".format(d, err))
                d = pd.NaT
        else:
            d = pd.NaT
        return d

    # The parser used on the date features
    def parse_date(self, date_feature):
        """
        Parses date features in YYMM format, fixes input errors
        and aligns datetime64 dates with a reference date
        """
        try:
            date_feature = pd.to_datetime(
                date_feature, format="%y%m", errors="coerce").map(self.fix_century)
        except Exception as e:
            message = "Failed to parse date array {}.\nReason: {}".format(
                date_feature, e)
            logger.error(message)
            raise RuntimeError(message)
        return date_feature


class DeltaTime(DateHandler, NamedFeatureTransformer):
    """Computes the duration between a date and a reference date in months.

    Parameters:
    -----------

    reference_date: A datetimelike
    unit: ['months', 'years']
    """

    def __init__(self, reference_date, unit='months', suffix=True):
        self.reference_date = reference_date
        super().__init__(self.reference_date)
        self.feature_suffix = "_DELTA_" + unit.upper()
        self.unit = unit

    def get_duration(self, target):
        if not pd.isna(target):
            delta = relativedelta.relativedelta(
                self.reference_date, target)
            if self.unit.lower() == 'months':
                duration = (delta.years * 12) + delta.months
            elif self.unit.lower() == 'years':
                duration = delta.years + 1
        else:
            logger.info("Failed to calculate time delta. Dates: {} and {}."
                        .format(target, self.reference_date))
            duration = np.nan
        return duration

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        # We need to ensure we have datetime objects.
        # The dateparser has to return Int64 to work with sklearn, so
        # we need to recast here.
        X_trans = pd.DataFrame()

        for f in X.columns:
            feature_name = f + self.feature_suffix
            try:
                try:
                    target = self.parse_date(X[f])
                except Exception as e:
                    raise e
                X_trans[feature_name] = target.map(self.get_duration)
            except Exception as e:
                logger.error("Failed to transform '{}' on feature {} for reason {}".format(
                    self.__class__.__name__, f, e))
                raise e
        self.feature_names = X_trans.columns.values.tolist()
        return X_trans


class MonthsToDonation(DateHandler, NamedFeatureTransformer):
    """ Calculates the elapsed months from sending the promotion
        to receiving a donation.
        The mailings usually were sent out over several months
        and in some cases, the donation is recorded as occurring
        before the mailing. In these cases, the sending date was
        probably not recorded correctly for the example in question.
        As a consequence, the first sending month will be used to
        calculate the time delta.
    """

    def __init__(self, reference_date=pd.datetime(1998, 6, 1)):
        self.reference_date = reference_date
        super().__init__(self.reference_date)

    def fit(self, X, y=None):
        return self

    def calc_diff(self, row):
        ref = row[0]
        target = row[1]

        if not pd.isna(ref) and not pd.isna(target):
            try:
                duration = relativedelta.relativedelta(target, ref).years * 12
                duration += relativedelta.relativedelta(target, ref).months

            except Exception as e:
                logger.error("Failed to calculate time delta. "
                             "Dates: {} and {}\nMessage: {}"
                             .format(row[0], row[1], e))
                duration = np.nan
        else:
            duration = np.nan
        return duration

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = list()

        X_trans = pd.DataFrame(index=X.index)
        for i in range(3, 25):
            try:
                feat_name = "MONTHS_TO_DONATION_" + str(i)
                send_date = X.loc[:, ["ADATE_" + str(i)]]
                recv_date = X.loc[:, ["RDATE_" + str(i)]]
            except KeyError as e:
                # One of the features is not there, can't compute the delta
                logger.info("Missing feature for MONTHS_TO_DONATION_{}.".format(i))
                continue

            try:
                try:
                    send_date = self.parse_date(send_date.squeeze())
                    send_date.loc[:] = send_date.min()
                except Exception as e:
                    raise e
                try:
                    recv_date = self.parse_date(recv_date.squeeze())
                except RuntimeError as e:
                    raise e
                diffs = pd.concat([send_date, recv_date], axis=1).agg(self.calc_diff, axis=1)
                X_trans = X_trans.join(pd.DataFrame(
                    diffs, columns=[feat_name], index=X_trans.index), how="inner")
                self.feature_names.extend([feat_name])
            except Exception as e:
                logger.error("Failed to transform '{}' "
                             "on feature {} for reason {}"
                             .format(feat_name,
                                     self.__class__.__name__, e))
                raise e
        return X_trans


class Hasher(NamedFeatureTransformer):

    def __init__(self, verbose=0, n_components=8, cols=None, drop_invariant=False, hash_method='md5'):
        super().__init__()
        self.verbose = verbose
        self.n_components = n_components
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.hash_method = hash_method
        self.he = HashingEncoder(verbose=self.verbose,
                                 n_components=self.n_components,
                                 cols=self.cols,
                                 drop_invariant=self.drop_invariant,
                                 return_df=True,
                                 hash_method=self.hash_method)

    def fit(self, X, y=None):
        self.he.fit(X, y)
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        features = X.columns.values.tolist()
        X_trans = self.he.transform(X, y)
        generated_features = self.he.get_feature_names()
        self.feature_names = [
            f + "_" + g for f in features for g in generated_features]
        X_trans.columns = self.feature_names
        return X_trans


class CategoricalImputer(NamedFeatureTransformer):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        X_trans = X.fillna(X.mode().iloc[0])
        self.feature_names = X_trans.columns.values.tolist()
        return X_trans


class NumericImputer(BaseEstimator):

    def __init__(self, n_iter=5, initial_strategy="median",
                 random_state=Config.get("random_seed"), verbose=0):
        super().__init__()
        self.n_iter = n_iter
        self.initial_strategy = initial_strategy
        self.random_state = random_state
        self.verbose = verbose
        self.feature_names = None

        self.imp = IterativeImputer(n_iter=self.n_iter,
                                    initial_strategy=self.initial_strategy,
                                    random_state=self.random_state,
                                    verbose=self.verbose)

    def fit(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        try:
            self.imp.fit_transform(X.values, y)
        except Exception as e:
            raise e
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        self.feature_names = X.columns.values.tolist()
        try:
            X_trans = self.imp.fit_transform(X.values)
        except Exception as e:
            raise e
        X_trans = pd.DataFrame(data=X_trans, columns=X.columns, index=X.index)
        return X_trans

    def fit_transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        self.feature_names = X.columns.values.tolist()
        try:
            X_trans = self.imp.fit_transform(X.values, y)
        except Exception as e:
            raise e
        X_trans = pd.DataFrame(data=X_trans, columns=X.columns, index=X.index)
        return X_trans

    def get_feature_names(self):
        if isinstance(self.feature_names, list):
            return self.feature_names
        else:
            raise(ValueError("Transformer {} has to be transformed first, cannot return feature names.".format(
                self.__class__.__name__)))


class TargetImputer(NamedFeatureTransformer):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.feature_names = ['TARGET_B']
        return self

    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))

        def set_true_if_donated(example):
            donated = None
            if pd.isna(example['TARGET_B']):
                donated = 1 if example['TARGET_D'] > 0.0 else 0
            else:
                donated = example['TARGET_B']
            return donated

        X_trans = pd.DataFrame(index=X.index, columns=['TARGET_B'], dtype="int64")

        X_trans['TARGET_B'] = X.agg(set_true_if_donated, axis=1)

        return X_trans


class ZipToCoords(NamedFeatureTransformer):

    def __init__(self):
        super().__init__()
        self.app_id = "ZJBxigwxa1QPHlWrtWH6"
        self.app_code = "OJBun02aepkFbuHmYn1bOg"
        try:
            with open(pathlib.Path(Config.get("data_dir"), "zip_db.pkl"), "rb") as zdb:
                self.locations = pickle.load(zdb)
        except Exception:
            zip_db = pd.read_csv(pathlib.Path(Config.get("data_dir"), "zipcodes2018.txt"))
            zip_db.columns = ["zip", "ZIP_latitude", "ZIP_longitude"]
            self.locations = zip_db.set_index("zip").to_dict('index')

    def _do_geo_query(self, q):
        geolocator = Here(app_id="ZJBxigwxa1QPHlWrtWH6", app_code="OJBun02aepkFbuHmYn1bOg")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.01, max_retries=4)
        try:
            return geolocator.geocode(query=q, exactly_one=True)
        except GeocoderTimedOut:
            return self._do_geo_query(q)

    def _get_location(self, example):
        if example.ZIP:
            zip = str(example.ZIP).rjust(5, '0')
            q = {'postalcode': zip, 'state': example.STATE}
            location = self._do_geo_query(q)
            if location:
                loc = {'ZIP_latitude': location.latitude, 'ZIP_longitude': location.longitude}
            else: 
                logger.info("Transformer {}: No location found for zip {} in state {}. Setting to 0, 0"
                            .format(self.__class__.__str__, zip, example.STATE))
                loc = {'ZIP_latitude': 0, 'ZIP_longitude': 0}
        else:
            print("Transformer {}: ZIP is NaN, setting location to NaN as well."
                  .format(self.__class__.__str__))
            loc = {'ZIP_latitude': np.nan, 'ZIP_longitude': np.nan}
        return loc

    def _extract_coords(self, example):
        try:
            return self.locations[example.ZIP]
        except KeyError:
            if example.STATE in ["AA", "AE", "AP"]:  # military zip, no coords available
                self.locations[example.ZIP] = {'ZIP_latitude': 38.8719, 'ZIP_longitude': 77.0563}
            else:
                try:
                    loc = self._get_location(example)
                    self.locations[example.ZIP] = loc
                except Exception as e:
                    logger.info("Transformer {}: Failed to retrieve missing zip. Reason: {}"
                                .format(self.__class__.__str__, e))
            return self.locations[example.ZIP]

    def fit(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        self.feature_names = ["ZIP_latitude", "ZIP_longitude"]
        return self

    def transform(self, X, y=None):
        X_trans = pd.DataFrame(index=X.index)
        X_trans = X.apply(self._extract_coords, axis=1, result_type="expand")
        try:
            with open(pathlib.Path(Config.get("data_dir"), "zip_db.pkl"), "wb") as zdb:
                pickle.dump(self.locations, zdb)
        except Exception as e:
            logger.warning("Failed to store updated zipcode database.")
        return X_trans


class ZeroVarianceSparseDropper(NamedFeatureTransformer):
    """
    Transformer to identify zero variance and optionally low variance features
    for removal. Data-type agnostic. It operates on the unique values and their counts.
    This works similarly to the R caret::nearZeroVariance function.
    Derived from: https://github.com/philipmgoddard/pipelines/blob/master/custom_transformers.py

    Params:
    -------
    - near zero     False: remove only zero var.
                    True: remove near zero as well
    - freq_cut      cutoff frequency ratio of most frequent
                    to second most frequent
    - unique_cut    cutoff for percentage unique values
    """
    def __init__(self, near_zero=True,
                 freq_cut=95 / 5,
                 unique_cut=0.1,
                 sparse_cut=0.1,
                 override=[]):
        self.near_zero = near_zero
        self.freq_cut = freq_cut
        self.unique_cut = unique_cut
        self.sparse_cut = sparse_cut
        self.override = set(override)
        self.feature_names = None
        self._dropped = []

    def fit(self, X, y=None):
        self.zero_var = []
        self.near_zero_var = []
        n_obs = X.shape[0]

        sparse_cols = [c for c in X.columns
                       if X[c].count() / n_obs <= self.sparse_cut]

        for feat, series in X.iteritems():
            val_count = series.value_counts(normalize=True)
            if len(val_count) <= 1:
                self.zero_var.append(feat)
                self.near_zero_var.append(feat)
                continue
            freq_ratio = val_count.values[0] / val_count.values[1]
            unq_percent = len(val_count) / n_obs
            if (unq_percent < self.unique_cut) and (freq_ratio > self.freq_cut):
                self.near_zero_var.append(feat)
        self.near_zero_var = set(self.near_zero_var + sparse_cols) - self.override
        self.zero_var = set(self.zero_var + sparse_cols) - self.override
        self._dropped = self.near_zero_var if self.near_zero else self.zero_var

        return self

    def transform(self, X, y=None):
        if self.near_zero:
            X_trans = X.drop(self.near_zero_var, axis=1)
        else:
            X_trans = X.drop(self.zero_var, axis=1)
        self.feature_names = X_trans.columns.values.tolist()
