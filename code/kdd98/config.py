# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:37:40 2018

@author: Florian Hochstrasser
"""

import os
import pandas as pd
import seaborn as sns

APP_HOME = os.path.dirname(os.path.abspath(__file__))

class App:
    __conf = {
        "root_dir": APP_HOME + "/",
        "data_dir": "../data/",
        "hdf_store": "kdd_cup98_datastore.h5",
        "learn_file_name": "cup98LRN.txt",
        "learn_name": "kddCup98Learn",
        "validation_file_name": "cup98VAL.txt",
        "validation_name": "kddCup98Validation",
        "random_seed": 42,
        "reference_date": pd.datetime(1997, 6, 1),
        "color_palette": sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, dark=0.5),
        "color_palette_binary": sns.cubehelix_palette(2, start=.5, rot=-.75, reverse=True, dark=0.5),
        "color_map": sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True, reverse=True, dark=0.5),
        "color_map_diverging": sns.diverging_palette(10, 220, sep=80, n=20, as_cmap=True)

    }

    __setters = []

    @staticmethod
    def config(name):
        return App.__conf[name]

    @staticmethod
    def set(name, value):
        if name in App.__setters:
            App.__conf[name] = value
        else:
            raise NameError("This config value is not settable.")
