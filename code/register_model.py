
backend = "r"  # r or python

# Libraries
from azureml.core import Workspace
from azureml.core.model import Model

# Set Azure Workspace
ws = Workspace.from_config("code/config_ws.json")
print(ws.name, ws.location, ws.resource_group, ws.location, sep='\t')

# Register model
model_file_name = 'model.RData' if backend == "r" else 'productive.pkl'
model = Model.register(workspace=ws, model_path="data/" + model_file_name,  model_name='titanic_pipeline')
for m in Model.list(ws, name='titanic_pipeline'):
    print(m.name, m.version)



