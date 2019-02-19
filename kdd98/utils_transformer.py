import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logging.basicConfig(filename=__name__+'.log', level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_feature_names_from_pipeline(pipeline):
    return [f[f.find('__')+2:] for f in pipeline.get_feature_names()]


def get_feature_names_from_transformer_collection(collection):
    try:
        # We have a pure column transformer. Concatenate the feature names
        names= [t[t.find('__')+2:] for t in collection.get_feature_names()]
    except AttributeError:
        # We have at least one pipeline in the collection. So do it manually
        names = []
        if not isinstance(collection, Pipeline):
            for t in collection.transformers:
                if isinstance(t[1], Pipeline):
                    # We get the feature names from the last step
                    names.append(t[1].named_steps[t[1].steps[-1]
                                                  [0]].get_feature_names())
                else:
                    names.append(t[1].get_feature_names())
        elif isinstance(collection, Pipeline):
            names.append(
                collection.named_steps[collection.steps[-1][0]].get_feature_names())
    return names


def update_df_with_transformed(df_old, new_features, transformer, drop=[], new_dtype=None):
    feat_names = get_feature_names_from_transformer_collection(transformer)
    transformed_df = pd.DataFrame(data=new_features, columns=feat_names,
                                  index=df_old.index)
    if new_dtype:
        transformed_df = transformed_df.astype(new_dtype)
    if all(f in df_old.columns.values.tolist() for f in feat_names):
        df_old[feat_names] = transformed_df
        df_new = df_old
    else:
        to_merge = [f for f in feat_names if f not in df_old.columns.values.tolist()]
        to_replace = [f for f in feat_names if f in df_old.columns.values.tolist()]
        df_old[to_replace] = transformed_df[to_replace]
        df_new = df_old.merge(transformed_df[to_merge], on=df_old.index.name)
    if len(drop) > 0:
        df_new.drop(drop, axis=1, inplace=True)
    return df_new
