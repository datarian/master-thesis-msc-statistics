# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:44 2018

@author: Florian Hochstrasser
"""

import os
import pandas as pd
import tools as etls
import dicts as edcts

from config import App

class TidyDataset:
    """
    Represents a tidy dataset for either training or test data of the kdd cup 1998.
    This class is recommended to load ready-to-work with data. Expects input data
    as distributed on UCI's machine learning repository.
    """
    data_path = App.config("data_dir")

    def __init__(self, file_name_csv=None, file_name_hdf=None):
        self.raw_data_file = file_name_csv
        self.hdf_data_file = file_name_hdf
        self.raw_data = None
        self.hdf_data = None

    def get_raw_datafile_path(self):
        """ Return relative path to csv data file"""
        return self.data_path+self.raw_data_file

    def read_csv_data(self):
        """ Read in csv data. """
        self.raw_data = pd.read_csv(self.data_path+self.raw_data_file,
                                    parse_dates=[0, 7],
                                    date_parser=etls.four_digit_date_parser,
                                    na_values=edcts.NA_CODES,
                                    dtype=edcts.DTYPE_CATEGORICAL,
                                    low_memory=False, # needed for mixed type columns
                                    memory_map=True # load file in memory
                                    )

    def process_raw(self, save_hdf=True):
        """
        Processes the raw csv import (splitting up multivalue variables and such)
        """

    def load_hdf(self):
        """ Load an existing processed file """

    def get_data(self):
        """
        Returns processed data ready for further analysis. Attempts to load
        hdf, if that's not available, reads csv and processes data first.
        """
        if not self.hdf_data:
            if os.path.isfile(self.data_path+self.hdf_data_file):
                # The data has been processed before, load hdf and return data
                self.load_hdf()
            elif self.raw_data:
                # The raw data is available, process, save hdf and return
                self.process_raw()
            else:
                self.read_csv_data()
                self.process_raw()

        return self.hdf_data
