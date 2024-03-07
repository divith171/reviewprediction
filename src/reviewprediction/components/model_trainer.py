import pandas as pd
import os
from reviewprediction import logger
from xgboost import XGBClassifier
import json
import joblib
import numpy as np
from reviewprediction.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):


        # Define the file paths (replace with your actual paths)
        train_data = self.config.train_data_path
        test_data= self.config.test_data_path

        # Load the JSON data
        try:
            with open(train_data, "r") as f:
                train_data = json.load(f)
            with open(test_data, "r") as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print("Error: JSON files not found at specified paths.")
        else:
            print("Train and test data loaded from JSON files.")

        # Extract features and labels
        X_train_final = []
        y_train = []
        for item in train_data:
            X_train_final.append(item["features"])
            y_train.append(item["label"])

        X_test_final = []
        y_test = []
        for item in test_data:
            X_test_final.append(item["features"])
            y_test = [item["label"] for item in test_data]  # Shortened list comprehension

        # Convert features back to NumPy arrays (if needed)
        X_train = np.array(X_train_final)
        X_test = np.array(X_test_final)

        # Define and train the XGBClassifier model
        model_XGB = XGBClassifier(n_estimators=self.config.n_estimators, min_child_weight=self.config.min_child_weight, max_depth=self.config.max_depth, learning_rate=self.config.learning_rate)
        model_XGB.fit(X_train, y_train)


        joblib.dump(model_XGB, os.path.join(self.config.root_dir, self.config.model_name))