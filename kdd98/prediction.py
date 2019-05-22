import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PowerTransformer


class Kdd98ProfitEstimator(BaseEstimator):
    """
    Estimates expected profit E(P) = \sum_{i=1}^n \hat{y}_{b,i} * \hat{y}_{d,i} * \alpha^*

    \hat{y}_{b,i} is the predicted probability of donating
    \hat{y}_{d,i} is the predicted donation amount (Box-Cox transformed)
    \alpha^* is a correction factor

    \hat{y,d} is predicted using X_D = {X | TARGET_B =1}. Because this is a non-random sample,
    a correction in the form of \alpha^* has to be applied. \alpha^* is calculated using the

    """

    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor
        self.target_transformer = PowerTransformer(
            method="box-cox", standardize=True)

    def _make_2d_array(self, y):
        y = np.array(y).reshape(-1, 1)
        return y

    def _filter_data_for_donations(self, X, y):
        self.mask = y.TARGET_B.astype("int").astype("bool")
        X_d = X.loc[self.mask, :].values
        y_d = y.TARGET_D[self.mask].values
        return (X_d, y_d)

    def _optimize_alpha(self, y_b_predict, y_d_predict, y_true):
        """
        Finds \alpha* using cubic splines. The derivative of
        the splines is taken and the roots calculated. The roots are
        then filtered to remove discontinuity points. The filtered set of roots
        are the candidates for \alpha*.
        Next, profit is calculated by interpolating the cubic spline
        at these candidate locations and the candidate leading to highest
        profit is chosen as \alpha*.
        """
        profit_data = self._generate_profit_for_alphas(
            y_b_predict, y_d_predict, y_true)

        s = CubicSpline(profit_data.alpha.values, profit_data.prediction.values)
        ds = s.derivative()

        roots = ds.roots()
        candidates = roots[~np.isnan(roots)]

        alpha_star = candidates[np.argmax(s(candidates))]

        return alpha_star

    def _generate_profit_for_alphas(self, y_b_predict, y_d_predict, y_true, u=0.68):
        """
        Generates profit estimates for a grid of alpha values.
        """
        alpha_grid = np.linspace(0.0, 1., 10000)

        data = [{"alpha": a,
                 "prediction": self._pi_alpha(y_b_predict, y_d_predict, y_true, alpha=a)}
                for a in alpha_grid]
        
        return pd.DataFrame(data)

    def _pi_alpha(self, y_b_predict, y_d_predict, y_true, alpha=1.0, u=0.68):
        """
        Calculates the net profit for a given optimization parameter alpha.
        This function is solely used to help find the optimal parameter alpha*.

        Params
        ------
        y_b_predict: Probability estimates for wheter an example donates.
        y_d_predict: Predicted donation amount, transformed with a power-transform
        y_d_true:    True donation amount
        alpha:       Optimization parameter
        u:           Unit cost, defaults to 0.68 $ US

        Returns
        -------
        profit A list with the predicted profits
        """

        # we transform the unit cost with the same
        # power transformer used for the target transformation.
        u_trans = self.target_transformer.transform(
            np.array(u).reshape(-1, 1)).ravel()[0]

        # The indicator function used to determine if an example is predicted to yield a profit.
        indicator = (y_b_predict * np.exp(y_d_predict) * alpha > np.exp(u_trans)).ravel()

        # We subtract the unit cost from
        # true donation amounts to get true profit per example.
        true_profit = y_true - u

        # We filter the true profit and return the sum
        return np.sum(true_profit[indicator])

    def fit(self, X, y):

        assert(isinstance(X, pd.DataFrame))

        # Fit classifier and predict donation probability
        self.classifier.fit(X, y.TARGET_B.values.astype("int"))
        # Predictions are returned as tuples (p_class_0, p_class_1)
        # We are only interested in p_class_1
        y_b_predict = self.classifier.predict_proba(X)[:,1]

        X_d, y_d = self._filter_data_for_donations(X, y)

        # Transform y_d before model training, also training the target transformer
        y_d_trans = self.target_transformer.fit_transform(
            self._make_2d_array(y_d)).ravel()

        # Fit regressor to predict donation amounts on the transformed data.
        self.regressor.fit(X_d, y_d_trans)

        y_d_predict = self.regressor.predict(X)

        self.alpha_star = self._optimize_alpha(
            y_b_predict, y_d_predict, y.TARGET_D.values)

    def predict(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_b_predict = self.classifier.predict_proba(X)[:,1]
        y_d_predict_transformed = self.regressor.predict(X)
        y_d_predict = self.target_transformer.inverse_transform(self._make_2d_array(y_d_predict_transformed))
        expected_profit = np.dot(y_b_predict, (y_d_predict * self.alpha_star))[0]

        return expected_profit
