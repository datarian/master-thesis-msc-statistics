import logging
import kdd98.data_handler as dh
from kdd98.transformers import ZeroVarianceSparseDropper

# Set up the logger
logging.basicConfig(filename=__name__ + '.log', level=logging.INFO)
logger = logging.getLogger(__name__)

provider = dh.KDD98DataProvider("cup98LRN_snip.txt")

numeric = provider.numeric_data
# clean = provider.clean_data

# a = set()
# for i in range(3, 25):
#     aa = clean.loc[:, "RFA_" + str(i)].tolist()
#     for r in aa:
#         if not isinstance(r, float):
#             if len(r) == 3:
#                 a.add(r[2])
# print(a)
