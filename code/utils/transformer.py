import pandas as pd
import numpy as np

# Set up the logger
import logging
logging.basicConfig(filename=__name__+'.log', level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_feature_names_from_pipeline(pipeline):
    return [f[f.find('__')+2:] for f in pipeline.get_feature_names()]

def get_feature_names_from_transformer_collection(collection):
    try:
        # We have a pure column transformer. Concatenate the feature names
        return [t[t.find('__')+2:] for t in collection.get_feature_names()]
    except AttributeError as e:
        # We have at least one pipeline in the collection. So do it manually
        names = []
        if not isinstance(collection, Pipeline):
            for t in collection.transformers:
                if isinstance(t[1], Pipeline):
                    # We get the feature names from the last step
                    names.append(t[1].named_steps[t[1].steps[-1][0]].get_feature_names())
                else:
                    names.append(t[1].get_feature_names())
        elif isinstance(collection, Pipeline):
            names.append(collection.named_steps[collection.steps[-1][0]].get_feature_names())
        return names

def update_df_with_transformed(df_old, new_features, transformer, drop=[],new_dtype=None):
    feat_names = get_feature_names_from_transformer_collection(transformer)
    transformed_df = pd.DataFrame(data=new_features, columns=feat_names,
                     index=df_old.index)
    if new_dtype:
        transformed_df = transformed_df.astype(new_dtype)
    df_old.update(transformed_df)
    if len(drop > 0):
        df_old.drop(drop, axis=1, inplace=True)
    return df_old
