# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

import numpy as np
import pandas as pd
import pickle
import os
from azureml.core.model import Model
import sys
sys.path.append("/var/azureml-app")
from init import *


def init():
    # Make the model global
    global d_pipelines

    # Get model path
    #model_path = "data/productive.pkl"
    model_path = Model.get_model_path(model_name='titanic_pipeline')
    print(model_path)
    print(os.path.isfile(model_path))

    # Load Model
    with open(model_path, "rb") as file:
        d_pipelines = pickle.load(file)
    print(d_pipelines)


# note you can pass in multiple rows for scoring
def run(input_json_string):
    #   pd.read_csv(dataloc + "titanic.csv").iloc[6:9, :].to_json("data/records.json", orient="index")
    #   input_json_string="data/records.json"
    print(input_json_string)

    try:
        # Read scoring data
        df = pd.read_json(input_json_string, orient="index")

        # Transform
        df = d_pipelines["pipeline_etl"].transform(df)

        # Fit
        yhat = scale_predictions(d_pipelines["pipeline_fit"].predict_proba(df),
                                 d_pipelines["pipeline_etl"].named_steps["undersample_n"].b_sample,
                                 d_pipelines["pipeline_etl"].named_steps["undersample_n"].b_all) \
            [:, 1].tolist()

        # you can return any data type as long as it is JSON-serializable
        return yhat

    except Exception as e:
        result = str(e)
        return result
