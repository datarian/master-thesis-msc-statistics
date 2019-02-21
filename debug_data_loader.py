import kdd98.data_loader as dl
from kdd98.data_loader import Cleaner
from kdd98.config import Config
from kdd98.transformers import DeltaTime, MonthsToDonation
from sklearn.compose import ColumnTransformer


provider = dl.KDD98DataLoader("cup98LRN_snip.txt")

preprocessed = provider.preprocessed_data

print(preprocessed.info())
