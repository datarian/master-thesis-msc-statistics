# -*- coding: utf-8 -*-
"""
Extract / Transform / Load script for the raw csv data.
Converts data types, creates categorical fields and generally tidies the data.
In the end, saves the TRAIN / test data in hdf5 files for later use.

@author: Florian Hochstrasser
"""

import pandas as pd

DATA_PATH = "../data/"
TRAIN = "cup98LRN.txt"
TEST = "cup98VAL.txt"