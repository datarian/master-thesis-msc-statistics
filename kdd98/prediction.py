import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PowerTransformer


class Kdd98ProfitEstimator(BaseEstimator):

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
        the splines is taken and the roots calculated. These
        are the candidates for \alpha*.
        Next, profit is calculated by interpolating the cubic spline
        at these candidate locations and the root leading to highest
        profit is chosen as \alpha*.
        """
        profit_data = self._generate_profit_for_alphas(
            y_b_predict, y_d_predict, y_true)

        s = CubicSpline(profit_data.alpha.values, profit_data.prediction.values)
        ds = s.derivative()

        roots = ds.roots()
        alpha_star = roots[np.argmax(s(roots))]

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
        y_b_predict: Predictios for wheter an example donates.
        y_d_predict: Predicted donation amount, transformed with a power-transform
        y_d_true:    True donation amount
        transformer: Transformation applied to y_true to normalize distribution
        alpha:       Optimization parameter
        u:           Unit cost, defaults to 0.68 $ US

        Returns
        -------
        profit A list with the predicted profits
        """

        # we transform the unit cost with the same power transformer used for the target transformation.
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

        # Fit classifier and predict donors
        self.classifier.fit(X, y.TARGET_B.values.astype("int"))
        y_b = self.classifier.predict(X)

        X_d, y_d = self._filter_data_for_donations(X, y)

        # Transform y_d before model training, also training the target transformer
        y_d_trans = self.target_transformer.fit_transform(
            self._make_2d_array(y_d)).ravel()

        # Fit regressor to predict donation amounts
        self.regressor.fit(X_d, y_d_trans)

        y_d = self.regressor.predict(X)

        self.alpha_star = self._optimize_alpha(
            y_b, y_d, y.TARGET_D.values)

    def predict(self, X, y=None):
        y_b = self.classifier.predict(X)
        y_d = self.regressor.predict(X)
        profit = np.dot(y_b, (y_d * self.alpha_star))

        return profit
