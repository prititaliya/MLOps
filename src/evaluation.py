import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import mlflow
mlflow.set_tracking_uri("http://ec2-3-15-213-179.us-east-2.compute.amazonaws.com:5000/")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test data.
    Parameters:
    model (XGBClassifier): The trained model to evaluate.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The test labels.
    Returns:
    dict: A dictionary containing the evaluation metrics.
    """
    try:
        logging.info("Starting model evaluation")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Classification Report: {report}")
        print(f"Confusion Matrix: \n{conf_matrix}")
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1-score", report['1']['f1-score'])
        return {
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        }
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        return None
def load_model(model_path="/Users/jatin/Desktop/MLOps/pipelines/model/xgboost_model.pkl"):
    """Load the trained model from a file.
    Parameters:
    model_path (str): The path to the model file.
    Returns:
    XGBClassifier: The loaded XGBoost model.
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None
def save_evaluation_results(results):
    """Save the evaluation results to a file.
    Parameters:
    results (dict): The evaluation results to save.
    """
    try:
        with open( "metrics.json", "w") as f:
            json.dump(results, f)
        logging.info(f"Evaluation results saved successfully to metrics.json")
    except Exception as e:
        logging.error(f"Error saving evaluation results to metrics.json: {e}")
def main():
    data = pd.read_csv("/Users/jatin/Desktop/MLOps/pipelines/feature_engineering/engineered_data.csv")
    X_test = data.drop('sentiment', axis=1)
    y_test = data['sentiment']
    model = load_model()
    mlflow.set_experiment("XGBoost")
    with mlflow.start_run():
        if model is not None:
            evaluation_results = evaluate_model(model, X_test, y_test)
            if evaluation_results is not None:
                save_evaluation_results(evaluation_results)
                logging.info(f"Classification Report: {evaluation_results['classification_report']}")
                logging.info(f"Confusion Matrix: \n{evaluation_results['confusion_matrix']}")
if __name__ == "__main__":
    main()