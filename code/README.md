# Profit maximisation for direct marketing campaigns

A complete data analysis of the KDD Cup 1998 data set. Optimizing net revenue for direct marketing campaigns of a US veterans organisation.

This repository contains a utility package (``kdd98``) as well as a collection of jupyter notebooks for reproducing the analysis.

## Package kdd98

A utility package providing data for analysis.

Data comes from the KDD Cup 1998, available at https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1998+Data

### What the package does

* Download raw data
* Apply transformations:
    - Preprocessing (input errors, usable data types)
    - Make all numeric (categorical encoding)
    - Imputation (using IterativeImputer)
    - Feature extraction using Boruta

The tranformations can be learned on the LRN data set, then applied on the VAL data set later, ensuring consistently transformed data.

### Usage

```{python}
from kdd98 import data_handler as dh

provider = dh.KDD98DataProvider(["cup98LRN.txt", "cup98VAL.txt"])

# Get data at several intermediate steps as necessary
provider.raw_data
provider.preprocessed_data
provider.numeric_data
provider.imputed_data
provider.all_relevant_data
```

### Installation

```
python setup.py install #

python setup.py bdist_conda # for a conda package
```

#### Project structure

After installing the package, several folders and files are created in the working directory when using it:

* ``data`` contains the data files and documentation of the original KDD Cup 1998
* ``data/data_frames`` contains pickled pandas df's with data at the corresponding transformation step
* ``models/internal`` contains persisted transformers for data set transformations.
* ``out.log`` reports transformation progress / potential problems

**Please note:**

* Do *not* touch the `models/` folder between fitting to LRN and transforming VAL data sets. The persisted models are necessary for the correct functioning of the packe.
* Upgrading python most likely will invalidate your persisted models and transformed data sets. After an update, clean everything under `models/` and `data/data_frames/` 

Additionally, this repository contains:

* notebooks: The jupyter notebooks to recreate the whole analysis
* kdd98/ is the source of the python package containing data handler and custom transformers supporting the analysis


## Report

The current state of the report's master branch is at: [https://datarian.github.io/master-thesis-report/](https://datarian.github.io/master-thesis-report/)

The repository containing the bookdown report files is at: https://github.com/datarian/master-thesis-report
