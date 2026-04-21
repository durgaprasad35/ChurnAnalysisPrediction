import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score

from src.exception import CustomException

from sklearn.metrics import accuracy_score , precision_score,recall_score,confusion_matrix
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) 

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)  
            cm_mat = confusion_matrix(y_test, y_test_pred)
            
            report[model_name] = test_model_score

            print(f"{model_name}: Train Accuracy = {train_model_score:.4f}, Test Accuracy = {test_model_score:.4f}")
            print(cm_mat)



        return report

    except Exception as e:
        raise CustomException(e, sys) 