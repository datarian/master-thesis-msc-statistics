# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:37:40 2018

@author: Florian Hochstrasser
"""

import os
import pathlib

import pandas as pd
import seaborn as sns

PKG_HOME = pathlib.Path(__file__).resolve().parent

__all__ = ['Config']


class Config:
    __conf = {
        "root_dir": pathlib.Path(PKG_HOME.resolve().parent),
        "data_dir": pathlib.Path(PKG_HOME.resolve().parent, "data"),
        "cache_dir": pathlib.Path(PKG_HOME.resolve().parent, "cache"),
        "download_files": ['cup98lrn.zip', 'cup98val.zip', 'cup98doc.txt', 'cup98dic.txt', 'instruct.txt', 'valtargt.readme', 'valtargt.txt', 'readme'],
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup98-mld/epsilon_mirror/",
        "hdf_store": "kdd_cup98_datastore.h5",
        "learn_file_name": "cup98LRN.txt",
        "learn_test_file_name": "cup98LRN_snip.txt",
        "learn_raw_name": "kddCup98Learn_raw",
        "learn_clean_name": "kddCup98Learn_clean",
        "learn_preproc_name": "kddCup98Learn_preproc",
        "validation_file_name": "cup98VAL.txt",
        "validation_raw_name": "kddCup98Validation_raw",
        "validation_clean_name": "kddCup98Validation_clean",
        "validation_preproc_name": "kddCup98Validation_preproc",
        "model_store": pathlib.Path(PKG_HOME.resolve().parent, "models"),
        "random_seed": 42,
        "reference_date": pd.datetime(1997, 6, 1),
        "color_palette": sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, dark=0.5),
        "color_palette_binary": sns.cubehelix_palette(2, start=.5, rot=-.75, reverse=True, dark=0.5),
        "color_map": sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True, reverse=False, dark=0.5),
        "color_map_diverging": sns.diverging_palette(10, 220, sep=80, n=20, as_cmap=True)
    }

    __setters = ["data_dir"]

    @staticmethod
    def get(*keys):
        """
        Returns one or more config keys either as a single object
        or a list of objects.
        """
        vals = [Config.__conf[val] for val in keys]
        return vals[0] if len(vals) == 1 else vals

    @staticmethod
    def set(key, value):
        if key in Config.__setters:
            Config.__conf[key] = value
        else:
            raise NameError("This config value is not settable.")
