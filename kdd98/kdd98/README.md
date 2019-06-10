# Kdd98 package

This package essentially provides the data set for the KDD98-CUP as a pandas dataframe with minimal transformations applied. Date and categorical features are cast to the correct types and the cast date features fixed.

## DataLoader

The *DataLoader* class holds various dictionaries for features of special interest, like nominal and ordinal features, dates, redundant features and so on. The specification of these features was taken from the data set documentation.
