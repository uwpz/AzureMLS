
backend = "r"  # r or python

# Libraries
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice, Webservice
import os

# --- Get Model -----------------------------------------------------------------------------------------
ws = Workspace.from_config("code/config_ws.json")
l_models = Model.list(ws, name='titanic_pipeline')
for m in l_models:
    print(m.name, m.version)
model = l_models[0]


# --- Create container -----------------------------------------------------------------------------------
# Define container contents
env = CondaDependencies()
if backend == "python":
    env.add_conda_package("pandas")
    env.add_conda_package("sklearn")
    env.add_conda_package("xgboost")
    env.add_conda_package("seaborn")
else:
    env.add_pip_package("rpy2")
with open("code/conda_file.yml", "w") as file:
    file.write(env.serialize_to_string())
print(env.serialize_to_string())

'''
# DOES NOT WORK
# Create Basis container
tmp = os.getcwd()
try:
    os.chdir(tmp + "/code")
    image_config = ContainerImage.image_configuration(
        execution_script="dummy.py",  # must be in cwd
        runtime="python",
        conda_file="conda_file.yml",
        docker_file="docker_file",
        dependencies=["init.py"] if backend == "python" else ["install_package.R", "hmsPM"])
    image = ContainerImage.create(workspace=ws,
                                  name="base-image",
                                  models=[],
                                  image_config=image_config)
    image.wait_for_creation(show_output=True)
    os.chdir(tmp)
except Exception as e:
    os.chdir(tmp)
    print(e)
    
# Create container image
tmp = os.getcwd()
try:
    os.chdir(tmp + "/code")
    image_config = ContainerImage.image_configuration(
        execution_script="score_model.py" if backend == "python" else "score_R_model.py",  # must be in cwd
        runtime="python",
        base_image="mlservice2042829589.azurecr.io/base-image:1")
    image = ContainerImage.create(workspace=ws,
                                  name="titanic-image-new",
                                  models=[model],
                                  image_config=image_config)
    image.wait_for_creation(show_output=True)
    os.chdir(tmp)
except Exception as e:
    os.chdir(tmp)
    print(e)
'''

# Create container image
tmp = os.getcwd()
try:
    os.chdir(tmp + "/code")
    image_config = ContainerImage.image_configuration(
        execution_script="score_model.py" if backend == "python" else "score_R_model.py",  # must be in cwd
        runtime="python",
        conda_file="conda_file.yml",
        docker_file="docker_file",
        dependencies=["init.py"] if backend == "python" else ["install_package.R", "hmsPM"])
    image = ContainerImage.create(workspace=ws,
                                  name="titanic-image",
                                  models=[model],
                                  image_config=image_config)
    image.wait_for_creation(show_output=True)
    print(image.image_build_log_uri)
    os.chdir(tmp)
except Exception as e:
    os.chdir(tmp)
    print(e)


# --- Create webservice ---------------------------------------------------------------------------
aci_config = AciWebservice.deploy_configuration(cpu_cores=1,
                                                memory_gb=1,
                                                tags={'sample name': 'AML 101'},
                                                description='This is a great example.')
service = Webservice.deploy_from_image(workspace=ws,
                                       name='titanic-webservice-new',  # crashes when already exist
                                       image=image,
                                       deployment_config=aci_config)
service.wait_for_deployment(show_output=True)
print(service.get_logs())

