import kdd98.data_loader as dl
from kdd98.data_loader import Cleaner
from kdd98.config import Config
from kdd98.transformers import DeltaTime, MonthsToDonation
from sklearn.compose import ColumnTransformer


provider = dl.KDD98DataLoader("cup98LRN_snip.txt")

cleaned = provider.clean_data

date_transformer = ColumnTransformer([
    ("date",
     DeltaTime(),
     ['MAXADATE'])
])

date = date_transformer.fit_transform(cleaned)

print(date)

don_hist_transformer = ColumnTransformer([
    ("months_to_donation",
     MonthsToDonation(),
     dl.PROMO_HISTORY_DATES+dl.GIVING_HISTORY_DATES
     )
])

donation_responses = don_hist_transformer.fit_transform(cleaned)

print(donation_responses)
