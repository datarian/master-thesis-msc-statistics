import logging
import kdd98.data_handler as dh
from kdd98.config import Config
from kdd98.prediction import Kdd98ProfitEstimator
import pickle
import pathlib

# Set up the logger
logging.basicConfig(filename=__name__ + '.log', level=logging.INFO)
logger = logging.getLogger(__name__)

learning_provider = dh.KDD98DataProvider("cup98LRN.txt")
test_provider = dh.KDD98DataProvider("cup98VAL.txt")
learning = learning_provider.all_relevant_data
test = test_provider.all_relevant_data

Config.set("model_store", pathlib.Path('/data/home/datarian/OneDrive/unine/Master_Thesis/ma-thesis-report/models'))

X_train = learning["data"]
y_train = learning["targets"]

X_test = test["data"]
y_test = test["targets"]

with open(pathlib.Path(Config.get("model_store"),"best_classifier.pkl"), "rb") as f:
    classifier = pickle.load(f)
    
with open(pathlib.Path(Config.get("model_store"),"best_regressor.pkl"), "rb") as f:
    regressor = pickle.load(f)

pe = Kdd98ProfitEstimator(classifier, regressor)

pe.fit(X_train, y_train)

prediction_unseen = pe.predict(X_test)

print(prediction_unseen)