import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PowerTransformer
from scipy.optimize import minimize, curve_fit


class Kdd98ProfitEstimator(BaseEstimator):
    
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor
        self.target_transformer = PowerTransformer(method="box-cox", standardize=True)

    def _make_2d_array(self, y):
        y = np.array(y).reshape(-1,1)
        return y
        
    def _filter_data_for_donations(self, X, y):
        mask = y.TARGET_B.astype("int").astype("bool")
        X_d = X.loc[mask,:].values
        y_d = y.TARGET_D[mask].values
        return (X_d, y_d)

    def _optimize_alpha(self,  y_d_predict, y_d, **kwargs):
        profit_data = self._generate_profit_for_alphas(y_d_predict, y_d, **kwargs)
        
        def gauss_pdf(x, mu, sigma):
            return 1/sigma*np.sqrt(2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)

        params = curve_fit(gauss_pdf, profit_data.alpha.values, profit_data.prediction.values)
        alpha_star = params[0][0]
        return alpha_star

    def _generate_profit_for_alphas(self, y_d_predict, y_d, u=0.68, n_iter_no_change=20):
        # Generates profit for a grid of alpha values.
        # Stops once there's no change (all predicted examples are selected)
        alpha_grid = np.linspace(0.0, 1., 1000)
        data = []
        profits = []
        for a in alpha_grid:
            # Check if profit converged, if it has, we selected all examples.
            # In that case, break and return the data
            if len(profits) > n_iter_no_change and profits[-1] != 0.0:
                if len(set(profits[-n_iter_no_change:])) == 1:
                    break
            profits.append(self._pi_alpha(y_d_predict, y_d, alpha=a))
            data.append({"alpha": a, "prediction": profits[-1]})
        return pd.DataFrame(data)


        
    def _pi_alpha(self, y_d_predict, y_d, alpha = 1.0, u=0.68):
        """
        Calculates the net profit for a given optimization parameter alpha.
        This function is solely used to help find the optimal parameter alpha*.
        
        Params
        ------
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
        u_trans = self.target_transformer.transform(np.array(u).reshape(-1,1)).ravel()[0]
        
        # The indicator function used to determine if an example is predicted to yield a profit.
        indicator = (np.exp(y_d_predict) * alpha > np.exp(u_trans)).ravel()
        
        true_profit = y_d - u
        
        # We filter the true profit and return the sum
        return np.sum(true_profit[indicator])
        
    def fit(self, X, y):
        
        # Fit classifier to predict donors
        y_b = y.TARGET_B.values.astype("int")
        self.classifier.fit(X, y_b)
        
        X_d, y_d = self._filter_data_for_donations(X, y)

        # Transform y_d before predicting, also training the target transformer
        y_d_trans = self.target_transformer.fit_transform(self._make_2d_array(y_d))

        # Fit regressor to predict donation amounts
        self.regressor.fit(X_d, y_d_trans)

        self.alpha_star = self._optimize_alpha(self.regressor.predict(X_d),y_d_trans)
        
    def predict(self, X, y=None):
        y_b = self.classifier.predict(X)
        
        X_d, y_d = self._filter_data_for_donations(X, y)
        y_d = self.regressor.predict(X) * self.alpha_star

        profit = np.dot(y_b,y_d)

        return profit
        
        