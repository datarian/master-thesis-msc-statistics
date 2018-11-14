import pandas as pd
import numpy as np

# Set up the logger
import logging
logging.basicConfig(filename=__name__+'.log', level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_feature_names_from_pipeline(pipeline):
    return [f[f.find('__')+2:] for f in pipeline.get_feature_names()]
