# -*- coding: utf-8 -*-
"""
Exploratory data analysis: Structure of the data,
summaries, value ranges, missing values

@author: Florian Hochstrasser
"""



def num_missing(column):
    """ Report number of missing values in a column"""
    return sum(column.isnull())

#Applying per column:
#axis=0 defines that function is to be applied on each column
#print(train_data.apply(num_missing, axis=0))
