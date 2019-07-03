import json
from azureml.core.model import Model
import rpy2.rinterface
import rpy2.robjects as robjects
import os

test_local = False  # for local test


# Run as container start
def init():
    # init rpy2
    rpy2.rinterface.initr()
    if test_local:
        robjects.r(".libPaths('C:/Users/pritzsche/Documents/R/win-library/3.5')")

    # Get model path
    if test_local:
        model_path = "data/model.RData"  # for local test
    else:
        model_path = Model.get_model_path(model_name='titanic_pipeline')
    print(model_path)
    print(os.path.isfile(model_path))

    # Load Model
    robjects.r("load('{model_path}')".format(model_path=model_path))

    # run init() function in R (if exists)
    robjects.r("if (exists('init', mode='function')) { init() }")

# init()

def run(input_json_string):
    if test_local:
        #import pandas as pd; pd.read_csv("data/titanic.csv").iloc[0:3, :].to_json("data/records.json", orient="index")
        with open("data/records.json") as file:
            input_json_string = file.read()

    try:
        result_vector = robjects.r("run('{0}')".format(input_json_string.replace("\\", "")))
        if len(result_vector) > 0:
            try:
                return json.loads(result_vector[0])
            except ValueError:
                return {"message": result_vector }
    except Exception as e:
        error = str(e)
        return {"error": error}

#  run("dummy")
