import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.components.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            ) 
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()

            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            logging.info("Starting model training and evaluation")

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            print("Model Report:", model_report)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 

            best_model = models[best_model_name]

            print(best_model_name)

            from imblearn.pipeline import Pipeline
            from imblearn.over_sampling import SMOTE

            pipeline = Pipeline([
                ("smote", SMOTE(random_state=42)),
                ("model", best_model)
            ])

           
            pipeline.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=pipeline
            ) 
            save_object(
                file_path="artifacts/label_encoder.pkl",
                obj=le
            )

            predicted = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)