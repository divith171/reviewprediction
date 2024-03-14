# reviewprediction

## Workflows

1. update config.yaml
2. update schema.yaml
3. update params.yaml
4. update the entity 
5. update the configuration manager in src config
6. update the components
7. update the pipeline
8. update the main.py
9. update the app.py



# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/divith171/reviewprediction
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.9 -y
```

```bash
conda activate venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```

### ml-flow

### dagshub

MLFLOW_TRACKING_URI=https://dagshub.com/divith171/reviewprediction.mlflow \
MLFLOW_TRACKING_USERNAME=divith171 \
MLFLOW_TRACKING_PASSWORD=f17ed95907e0d5c1e4b27b30354f6ed03187f1ba \
python script.py

run this as an env variable
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/divith171/reviewprediction.mlflow 
export MLFLOW_TRACKING_USERNAME=divith171 
export MLFLOW_TRACKING_PASSWORD=f17ed95907e0d5c1e4b27b30354f6ed03187f1ba 


```
