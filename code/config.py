# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:37:40 2018

@author: Florian Hochstrasser
"""

import os


class App:
    __conf = {
        "root_dir": os.getcwd()+"/",
        "data_dir": "../data/",
        "hdf_store": "kdd_cup98_datastore.h5",
        "learn_file_name": "cup98LRN.txt",
        "learn_name": "kddCup98Learn",
        "validation_file_name": "cup98VAL.txt",
        "validation_name": "kddCup98Validation",
        "random_seed": 42
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
