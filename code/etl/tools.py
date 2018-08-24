# -*- coding: utf-8 -*-
"""
Functions needed to treat difficult variable transformations.

@author: Florian Hochstrasser
"""

import pandas as pd
import numpy as np
from etl.dicts import BooleanRecodeConfig


def four_digit_date_parser(date):
    """ Formats YYMM dates as YYYY-MM-DD where DD is the first always. """
    if len(date) is 4:
        parsed_date=pd.to_datetime(date, format="%y%m")
    else:
        parsed_date=np.nan

    return parsed_date


def recode_booleans(data, recode_fields):
    """
    Recodes boolean columns. Specify the codes used in data and the affected
    columns through BooleanRecodeConfig objects.
    Expects a pandas data frame!
    """

    if not isinstance(recode_fields[0], BooleanRecodeConfig):
        raise TypeError("Expects a list of BooleanRecodeConfig.")

    # Ensure we have a data frame
    data = pd.DataFrame(data)

    def do_recode(recode_config):
        if not isinstance(recode_config, BooleanRecodeConfig):
            raise TypeError("Expects aBooleanRecodeConfig.")

        true_char = recode_config.value_mapping.get('t')
        false_char = recode_config.value_mapping.get('f')
        for field in recode_config.fields:
            data.loc[data.field not in [true_char, false_char]] = np.nan
            data.loc[data.field is false_char] = False
            data.loc[data.field is true_char] = True

    for recode in recode_fields:
        do_recode(recode)


def recode_promotion_history():
    """
    The promotion history data is aggregated
    """
