
from azureml.core import Workspace
import requests
import pandas as pd

pd.read_csv("data/titanic.csv").iloc[6:9, :].to_json("data/records.json", orient="index")

ws = Workspace.from_config("code/config_ws.json")
service = ws.webservices['titanic-webservice']

with open("data/records.json") as file:
    test_samples = file.read()

print(service.run(input_data=str(test_samples)))
print(requests.post(service.scoring_uri, test_samples, headers={'Content-Type': 'application/json'}).text)

service.delete()