# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import sys
import hashlib
import copy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
from util.utils_catenc import get_obj_cols, convert_input, get_generated_cols
import datetime
from dateutil import relativedelta
import pandas as pd
import logging
# Set up the logger
logging.basicConfig(filename=__name__+'.log', level=logging.WARNING)
logger = logging.getLogger(__name__)


__all__ = [
    'BooleanFeatureRecode',
    'MultiByteExtract',
    'ZipCodeFormatter',
    'ParseDates',
    'ComputeAge',
    'MonthsToDonation',
    'HashingEncoder',
    'OneHotEncoder',
    'OrdinalEncoder'
]


class MultiByteExtract(BaseEstimator, TransformerMixin):
    def __init__(self, field_names, impute=False, drop_orig=True):
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
        self.is_fitted = False

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.feature_names = list(X.columns)
        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        zip_series = X.ZIP.copy()
        zip_series = zip_series.str.replace('-', '').astype('category')
        return pd.DataFrame(zip_series).astype('int')

    def get_feature_names(self):
        if self.is_fitted:
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


class HashingEncoder(BaseEstimator, TransformerMixin):
    """A basic multivariate hashing implementation with configurable dimensionality/precision.

    The advantage of this encoder is that it does not maintain a dictionary of observed categories.
    Consequently, the encoder does not grow in size and accepts new values during data scoring
    by design.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    hash_method: str
        which hashing method to use. Any method from hashlib works.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = HashingEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 19 columns):
    col_0      506 non-null int64
    col_1      506 non-null int64
    col_2      506 non-null int64
    col_3      506 non-null int64
    col_4      506 non-null int64
    col_5      506 non-null int64
    col_6      506 non-null int64
    col_7      506 non-null int64
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(8)
    memory usage: 75.2 KB
    None

    References
    ----------
    .. [1] Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing for
    Large Scale Multitask Learning. Proc. ICML.

    """

    def __init__(self, verbose=0, n_components=8, cols=None, drop_invariant=False, return_df=True, hash_method='md5'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.n_components = n_components
        self.cols = cols
        self.hash_method = hash_method
        self._dim = None
        self.feature_names = list()
        self.is_fitted = False

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # first check the type
        X = convert_input(X)

        self._dim = X.shape[1]

        print(self.cols)
        print(X.columns)
        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [
                x for x in generated_cols if X_temp[x].var() <= 10e-5]

        # Build list of feature names
        for c in list(set(self.cols) - set(self.drop_cols)):
            self.feature_names.extend([c+"_"+str(i)
                                       for i in range(0, self.n_components)])
        self.is_fitted = True

        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # first check the type
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim, ))

        if not self.cols:
            return X

        X = self.hashing_trick(
            X, hashing_method=self.hash_method, N=self.n_components, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns:
        --------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!
        """

        if not self.is_fitted:
            raise ValueError(
                'Must fit data first. Affected feature names are not known before.')
        else:
            return self.feature_names

    @staticmethod
    def hashing_trick(X_in, hashing_method='md5', N=2, cols=None, make_copy=False):
        """A basic hashing implementation with configurable dimensionality/precision

        Performs the hashing trick on a pandas dataframe, `X`, using the hashing method from hashlib
        identified by `hashing_method`.  The number of output dimensions (`N`), and columns to hash (`cols`) are
        also configurable.

        Parameters
        ----------

        X_in: pandas dataframe
            description text
        hashing_method: string, optional
            description text
        N: int, optional
            description text
        cols: list, optional
            description text
        make_copy: bool, optional
            description text

        Returns
        -------

        out : dataframe
            A hashing encoded dataframe.

        References
        ----------
        Cite the relevant literature, e.g. [1]_.  You may also cite these
        references in the notes section above.
        .. [1] Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing
        for Large Scale Multitask Learning. Proc. ICML.

        """

        try:
            if hashing_method not in hashlib.algorithms_available:
                raise ValueError('Hashing Method: %s Not Available. Please use one from: [%s]' % (
                    hashing_method,
                    ', '.join([str(x) for x in hashlib.algorithms_available])
                ))
        except Exception as e:
            try:
                _ = hashlib.new(hashing_method)
            except Exception as e:
                raise ValueError('Hashing Method: %s Not Found.')

        if make_copy:
            X = X_in.copy(deep=True)
        else:
            X = X_in

        if cols is None:
            cols = X.columns.values

        def hash_fn(x):
            tmp = [0 for _ in range(N)]
            for val in x.values:
                if val is not None:
                    hasher = hashlib.new(hashing_method)
                    if sys.version_info[0] == 2:
                        hasher.update(str(val))
                    else:
                        hasher.update(bytes(str(val), 'utf-8'))
                    tmp[int(hasher.hexdigest(), 16) % N] += 1
            return pd.Series(tmp, index=new_cols)

        new_cols = ['col_%d' % d for d in range(N)]

        X_cat = X.reindex(columns=cols)
        X_num = X.reindex(
            columns=[x for x in X.columns.values if x not in cols])

        X_cat = X_cat.apply(hash_fn, axis=1)
        X_cat.columns = new_cols

        X = pd.merge(X_cat, X_num, left_index=True, right_index=True)

        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Onehot (or dummy) coding for categorical features, produces one feature per category, each binary.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'error', 'ignore' and 'impute', defaults to 'impute', which will impute the category -1. Warning: if
        impute is used, an extra column will be added in if the transform matrix has unknown categories. This can cause
        unexpected changes in the dimension in some cases.
    use_cat_names: bool
        if True, category values will be included in the encoded column names. Otherwise category
        indices will be used.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = OneHotEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 24 columns):
    CHAS_1     506 non-null int64
    CHAS_2     506 non-null int64
    CHAS_-1    506 non-null int64
    RAD_1      506 non-null int64
    RAD_2      506 non-null int64
    RAD_3      506 non-null int64
    RAD_4      506 non-null int64
    RAD_5      506 non-null int64
    RAD_6      506 non-null int64
    RAD_7      506 non-null int64
    RAD_8      506 non-null int64
    RAD_9      506 non-null int64
    RAD_-1     506 non-null int64
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(13)
    memory usage: 95.0 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, impute_missing=True, handle_unknown='impute', use_cat_names=False):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
        self.use_cat_names = use_cat_names
        self.is_fitted = False

    @property
    def category_mapping(self):
        return self.ordinal_encoder.category_mapping

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # first check the type
        X = convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        # Indicate no transformation has been applied yet

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [
                x for x in generated_cols if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # first check the type
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim, ))

        if not self.cols:
            return X if self.return_df else X.values

        X = self.ordinal_encoder.transform(X)

        X = self.get_dummies(X, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        # Now we can build the list of of new / transformed columns
        self.feature_names = X.columns
        self.is_transformed = True

        if self.return_df:
            return X
        else:
            return X.values

    def inverse_transform(self, X_in):
        """
        Perform the inverse transformation to encoded data.

        Parameters
        ----------
        X_in : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p: array, the same size of X_in

        """
        X = X_in.copy(deep=True)

        # first check the type
        X = convert_input(X)

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to inverse_transform data')

        X = self.reverse_dummies(X, self.cols)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError("Unexpected input dimension %d, the attribute drop_invariant should "
                                 "set as False when transform data" % (X.shape[1],))
            else:
                raise ValueError('Unexpected input dimension %d, expected %d' % (
                    X.shape[1], self._dim, ))

        if not self.cols:
            return X if self.return_df else X.values

        if self.impute_missing and self.handle_unknown == 'impute':
            for col in self.cols:
                if any(X[col] == -1):
                    raise ValueError("inverse_transform is not supported because transform impute "
                                     "the unknown category -1 when encode %s" % (col,))
        if not self.use_cat_names:
            for switch in self.ordinal_encoder.mapping:
                col_dict = {col_pair[1]: col_pair[0]
                            for col_pair in switch.get('mapping')}
                X[switch.get('col')] = X[switch.get('col')].apply(
                    lambda x: col_dict.get(x))

        return X if self.return_df else X.values

    def get_dummies(self, X_in, cols=None):
        """
        Convert numerical variable into dummy variables

        Parameters
        ----------
        X_in: DataFrame
        cols: list-like, default None
              Column names in the DataFrame to be encoded
        Returns
        -------
        dummies : DataFrame
        """

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values
            pass_thru = []
        else:
            pass_thru = [col for col in X.columns.values if col not in cols]

        bin_cols = []
        for col in cols:
            col_tuples = copy.deepcopy(
                [class_map['mapping'] for class_map in self.ordinal_encoder.mapping if class_map['col'] == col][0])
            if self.handle_unknown == 'impute':
                col_tuples.append(('-1', -1))
            for col_tuple in col_tuples:
                class_ = col_tuple[1]
                cat_name = col_tuple[0]
                if self.use_cat_names:
                    n_col_name = str(col) + '_%s' % (cat_name, )
                else:
                    n_col_name = str(col) + '_%s' % (class_, )

                X[n_col_name] = X[col] == class_
                bin_cols.append(n_col_name)

        X = X.reindex(columns=bin_cols + pass_thru)

        # convert all of the bools into integers.
        for col in bin_cols:
            X[col] = X[col].astype(int)

        return X

    def reverse_dummies(self, X, cols):
        """
        Convert dummy variable into numerical variables

        Parameters
        ----------
        X : DataFrame
        cols: list-like
              Column names in the DataFrame that be encoded

        Returns
        -------
        numerical: DataFrame

        """
        out_cols = X.columns.values

        for col in cols:
            col_list = [col0 for col0 in out_cols if str(
                col0).startswith(str(col))]
            # original column name plus underscore
            prefix_length = len(str(col))+1
            if self.use_cat_names:
                X[col] = 0
                for tran_col in col_list:
                    val = tran_col[prefix_length:]
                    X.loc[X[tran_col] == 1, col] = val
            else:
                value_array = np.array(
                    [int(col0[prefix_length:]) for col0 in col_list])
                X[col] = np.dot(X[col_list].values, value_array.T)
            out_cols = [col0 for col0 in out_cols if col0 not in col_list]

        X = X.reindex(columns=out_cols + cols)

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns:
        --------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!
        """

        if not self.is_transformed:
            raise ValueError(
                'Must transform data first. Affected feature names are not known before.')
        else:
            return self.feature_names


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features as ordinal, in one ordered feature.

    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
   mapping: list of dict
        a mapping of class to label to use for the encoding, optional.
        the dict contains the keys 'col' and 'mapping'.
        the value of 'col' should be the feature name.
        the value of 'mapping' should be a list of tuples of format (original_label, encoded_label).
        example mapping: [{'col': 'col1', 'mapping': [(None, 0), ('a', 1), ('b', 2)]}]
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'error', 'ignore' and 'impute', defaults to 'impute', which will impute the category -1.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = OrdinalEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null int64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null int64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(2)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """

    def __init__(self, verbose=0, mapping=None, cols=None, drop_invariant=False, return_df=True, impute_missing=True,
                 handle_unknown='impute'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.mapping = mapping
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
        self._dim = None
        self.feature_names = []
        self.is_fitted = False

    @property
    def category_mapping(self):
        return self.mapping

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # first check the type
        X = convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        _, categories = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )
        self.mapping = categories

        for switch in self.mapping:
            for cat in switch.get('mapping'):
                self.feature_names.extend([str(switch.get('col'))+"_"+str(cat[0])])

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [
                x for x in generated_cols if X_temp[x].var() <= 10e-5]
            for col in self.drop_cols:
                d = X_temp.columns.get_loc(col)
                self.feature_names.remove(d)
        self.is_fitted = True
        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

        Will use the mapping (if available) and the column list (if available, otherwise every column) to encode the
        data ordinarily.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # first check the type
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim,))

        if not self.cols:
            return X if self.return_df else X.values

        X, _ = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        return X if self.return_df else X.values

    def inverse_transform(self, X_in):
        """
        Perform the inverse transformation to encoded data.

        Parameters
        ----------
        X_in : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p: array, the same size of X_in

        """
        X = X_in.copy(deep=True)

        # first check the type
        X = convert_input(X)

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to inverse_transform data')

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError("Unexpected input dimension %d, the attribute drop_invariant should "
                                 "set as False when transform data" % (X.shape[1],))
            else:
                raise ValueError('Unexpected input dimension %d, expected %d' % (
                    X.shape[1], self._dim,))

        if not self.cols:
            return X if self.return_df else X.values

        if self.impute_missing and self.handle_unknown == 'impute':
            for col in self.cols:
                if any(X[col] == -1):
                    raise ValueError("inverse_transform is not supported because transform impute "
                                     "the unknown category -1 when encode %s" % (col,))

        for switch in self.mapping:
            col_dict = {col_pair[1]: col_pair[0]
                        for col_pair in switch.get('mapping')}
            X[switch.get('col')] = X[switch.get('col')].apply(
                lambda x: col_dict.get(x))

        return X if self.return_df else X.values

    @staticmethod
    def ordinal_encoding(X_in, mapping=None, cols=None, impute_missing=True, handle_unknown='impute'):
        """
        Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
        in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
        are assumed to have no true order and integers are selected at random.
        """

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                categories_dict = dict(switch.get('mapping'))
                column = switch.get('col')
                transformed_column = X[column].map(
                    lambda x: categories_dict.get(x))

                if impute_missing:
                    if handle_unknown == 'impute':
                        transformed_column.fillna(0, inplace=True)
                    elif handle_unknown == 'error':
                        missing = transformed_column.isnull()
                        if any(missing):
                            raise ValueError(
                                'Unexpected categories found in column %s' % column)

                try:
                    X[column] = transformed_column.astype(int)
                except ValueError as e:
                    X[column] = transformed_column.astype(float)
        else:
            mapping_out = []
            for col in cols:
                categories = [x for x in pd.unique(
                    X[col].values) if x is not None]
                categories_dict = {x: i + 1 for i, x in enumerate(categories)}

                mapping_out.append({'col': col, 'mapping': [
                                   (x[1], x[0] + 1) for x in list(enumerate(categories))]})

        return X, mapping_out

    def get_feature_names(self):
        if not self.is_fitted:
            raise ValueError("Estimator has to be fitted first.")
        else:
            return self.feature_names
