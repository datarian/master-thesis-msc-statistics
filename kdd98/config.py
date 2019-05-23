# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:37:40 2018

@author: Florian Hochstrasser
"""

import os
import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.colors as col

HOME = pathlib.Path.cwd().resolve()

__all__ = ['Config']


class Config:
    __conf = {
        "root_dir": pathlib.Path(HOME),
        "data_dir": pathlib.Path(HOME, "data"),
        "model_store": pathlib.Path(HOME, "models"),
        "model_store_internal": pathlib.Path(HOME, "models", "internal"),
        "df_store": pathlib.Path(HOME, "data", "data_frames"),
        "cache_dir": pathlib.Path(HOME, "cache"),
        "download_files": ['cup98lrn.zip', 'cup98val.zip', 'cup98doc.txt',
                           'cup98dic.txt', 'instruct.txt', 'valtargt.readme',
                           'valtargt.txt', 'readme'],
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup98-mld/epsilon_mirror/",
        "learn_file_name": "cup98LRN.txt",
        "learn_test_file_name": "cup98LRN_snip.txt",
        "learn_raw_name": "kddCup98Learn_raw",
        "test_raw_name": "kddCup98Test_raw",
        "learn_cleaned_name": "kddCup98Learn_cleaned",
        "test_cleaned_name": "kddCup98Test_cleaned",
        "learn_numeric_name": "kddCup98Learn_numeric",
        "test_numeric_name": "kddCup98Test_numeric",
        "learn_imputed_name": "kddCup98Learn_imputed",
        "learn_ar_name": "kddCup98Learn_all_relevant",
        "validation_file_name": "cup98VAL.txt",
        "validation_target_file_name": "valtargt.txt",
        "validation_raw_name": "kddCup98Validation_raw",
        "validation_cleaned_name": "kddCup98Validation_cleaned",
        "validation_numeric_name": "kddCup98Validation_numeric",
        "validation_imputed_name": "kddCup98Validation_imputed",
        "validation_ar_name": "kddCup98Validation_all_relevant",
        "random_seed": 42,
        "reference_date": pd.datetime(1997, 6, 1),
        "qual_palette": sns.husl_palette(8),
        "qual_palette_binary": sns.palettes.color_palette(['#f77189', '#39a7d0']), # 1st and 6th color from husl_palette
        "seq_palette": sns.cubehelix_palette(12, start=2.4, rot=0.5, gamma=0.9, hue=0.8, light=0.6, dark=0.2),
        "diverging_palette": sns.diverging_palette(204, 359, s=83, l=57, sep=10, n=12),
        "qual_color_map": col.LinearSegmentedColormap.from_list("husl", sns.husl_palette(8), N=8),
        "seq_color_map": sns.cubehelix_palette(256, start=2.4, rot=0.5, gamma=0.9, hue=0.8, light=0.6, dark=0.2, as_cmap=True),
        "diverging_color_map": sns.diverging_palette(204, 359, s=83, l=57, sep=10, n=12, as_cmap=True)
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

    @staticmethod
    def get_keys():
        return list(Config.__conf.keys())
