import kdd98.data_loader as dl
from kdd98.data_loader import Cleaner
from kdd98.config import Config


provider = dl.KDD98DataLoader("cup98LRN_snip.txt")

cleaned = provider.clean_data

print(cleaned.DOMAINSocioEconomic.cat.categories)
