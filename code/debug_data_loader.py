import logging
import kdd98.data_handler as dh
from kdd98.transformers import ZeroVarianceSparseDropper
from kdd98.config import Config
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from kdd98.transformers import Rescaler
from kdd98.prediction import Kdd98ProfitEstimator

# Set up the logger
logging.basicConfig(filename=__name__ + '.log', level=logging.INFO)
logger = logging.getLogger(__name__)

provider = dh.KDD98DataProvider("cup98LRN.txt")

#numeric = provider.numeric_data

#imputed = provider.imputed_data

all_relevant = provider.all_relevant_data

mlp_sampler = BorderlineSMOTE(random_state=Config.get("random_seed"))
mlp_scaler = Rescaler(transformer="ptrans")
svc_scaler = Rescaler(transformer="ptrans")

svc_classifier = SVC(
    class_weight="balanced",
    C=35,
    coef0=0.174,
    kernel="poly",
    degree=3,
    probability=True,
    gamma="auto",
    random_state=Config.get("random_seed"))



mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(50, 10,),
    alpha=0.5622,
    learning_rate_init=0.0842,
    early_stopping=True,
    random_state=Config.get("random_seed")
)

classifier = Pipeline([
    ("scaler", mlp_scaler),
    ("sampler", mlp_sampler),
    ("classifier", mlp_classifier)
])

regressor = SVR(C=72, degree=12, gamma=3.91)

X = all_relevant["data"].values
y_b = all_relevant["targets"].TARGET_B.values

mask = all_relevant["targets"].TARGET_B.astype("bool")
X_d = all_relevant["data"].loc[mask, :].values
y_d = all_relevant["targets"].TARGET_D[mask].values

classifier.fit(X,y_b)
regressor.fit(X_d,y_d)

pe = Kdd98ProfitEstimator(classifier, regressor)

pe.fit(all_relevant["data"], all_relevant["targets"])

profit = pe.predict(all_relevant["data"], all_relevant["targets"])

print(profit)