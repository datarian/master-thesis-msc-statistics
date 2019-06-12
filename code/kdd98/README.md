# Kdd98 package

Data provisioning for the kdd-cup 98.

Preprocessing and feature engineering are implemented in module `data_handler`.

Module `transformers` contains scikit-learn compatible data transformers.

Predictions can be made with module `prediction`, where a classifier and a regressor for the binary and continuous targets have to be passed to the class Kdd98ProfitEstimator.
