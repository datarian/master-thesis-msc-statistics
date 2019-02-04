# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:37:40 2018

@author: Florian Hochstrasser
"""

import os
import pathlib

import pandas as pd
import seaborn as sns

APP_HOME = pathlib.Path(__file__).resolve()

__all__ = ['App']

class App:
    __conf = {
        "root_dir": APP_HOME.parent,
        "data_dir": APP_HOME.parent / "data",
        "download_files": ['cup98lrn.zip', 'cup98val.zip', 'cup98doc.txt', 'cup98dic.txt', 'instruct.txt', 'valtargt.readme', 'valtargt.txt', 'readme'],
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup98-mld/epsilon_mirror/",
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
