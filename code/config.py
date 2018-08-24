# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:37:40 2018

@author: Florian Hochstrasser
"""

import os 

class App:
    __conf = {
            "root_dir": os.getcwd()+"\\",
            "data_dir": "..\\data\\",
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