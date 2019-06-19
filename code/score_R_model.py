import json
from azureml.core.model import Model
import rpy2.rinterface
import rpy2.robjects as robjects
import os


# Run as container start
def init():
    # init rpy2
    rpy2.rinterface.initr()
    #  robjects.r(".libPaths('C:/Users/pritzsche/Documents/R/win-library/3.4')")  # for local test

    # Get model path
    model_path = Model.get_model_path(model_name='titanic_pipeline')
    #model_path = "data/model.RData"   # for local test
    print(model_path)
    print(os.path.isfile(model_path))

    # Load Model
    robjects.r("load('{model_path}')".format(model_path=model_path))

    # run init() function in R (if exists)
    robjects.r("if (exists('init', mode='function')) { init() }")


def run(input_json_string):
    '''
    import pandas as pd; pd.read_csv("data/titanic.csv").iloc[0:3, :].to_json("data/records.json", orient="index")
    with open("data/records.json") as file:
        input_json_string = file.read()
    '''

    try:
        result_vector = robjects.r("run('{0}')".format(input_json_string.replace("\\", "")))
        if len(result_vector) > 0:
            try:
                return json.loads(result_vector[0])
            except ValueError:
                return {"message": result_vector }
    except Exception as e:
        error = str(e)
        return {"error": error }
