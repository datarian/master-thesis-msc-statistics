import pandas as pd
import numpy as np

# Set up the logger
import logging
logging.basicConfig(filename=__name__+'.log', level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_feature_names_from_pipeline(pipeline):
    return [f[f.find('__')+2:] for f in pipeline.get_feature_names()]


def update_df_with_transformed(df_old, new_features, transformer, drop=[],new_dtype=None):
    feat_names = [n[n.find('__')+2:]
                 for n in transformer.get_feature_names()]
    transformed_df = pd.DataFrame(data=new_features, columns=feat_names,
                     index=df_old.index)
    if new_dtype:
        transformed_df = transformed_df.astype(new_dtype)
    df_old.update(transformed_df)
    if len(drop > 0):
        df_old.drop(drop, axis=1, inplace=True)
    return df_old
