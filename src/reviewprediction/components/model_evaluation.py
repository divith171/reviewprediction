import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from urllib.parse import urlparse
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import json
from reviewprediction.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from reviewprediction.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/divith171/reviewprediction.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"]="divith171"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="f17ed95907e0d5c1e4b27b30354f6ed03187f1ba"


    
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred, average='weighted')  # Weighted F1-score for multi-class
        auc = roc_auc_score(actual, pred)
        return accuracy, f1, auc
    


    def log_into_mlflow(self):

        test_data = self.config.test_data_path
         # Load the JSON data
        try:
            
            with open(test_data, "r") as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print("Error: JSON files not found at specified paths.")
        else:
            print("Train and test data loaded from JSON files.")
        X_test_final = []
        y_test = []
        for item in test_data:
            X_test_final.append(item["features"])
            y_test = [item["label"] for item in test_data]  # Shortened list comprehension
        X_test = np.array(X_test_final)

        model = joblib.load(self.config.model_path)
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store= urlparse(mlflow.get_tracking_uri()).scheme



        with mlflow.start_run():

          predicted_qualities = model.predict(X_test)

          (accuracy, f1, auc) = self.eval_metrics(y_test, predicted_qualities)
        
          # Saving metrics as local
          scores = {"accuracy_score": accuracy, "f1 score": f1, "roc_auc_scor": auc}
          save_json(path=Path(self.config.metric_file_name), data=scores)
          mlflow.log_params(self.config.all_params)
          mlflow.log_metric("accuracy_score",accuracy)
          mlflow.log_metric("f1 score", f1)
          mlflow.log_metric("roc_auc_scor", auc)

         