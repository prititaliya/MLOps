import pandas as pd
import numpy as np
import logging
from pathlib import Path
from xgboost import XGBClassifier
import pickle
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import mlflow

def train_model(X_train, y_train):
    """Train an XGBoost model on the training data.
    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training labels.
    Returns:
    XGBClassifier: The trained XGBoost model.
    """
    try:
        logging.info("Starting model training")
        model = XGBClassifier()
        model.fit(X_train, y_train)
        mlflow.log_param("model_type", "XGBClassifier")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None
def save_model(model, output_dir="/Users/jatin/Desktop/MLOps/pipelines/model"):
    """Save the trained model to a file.
    Parameters:
    model (XGBClassifier): The trained model to save.
    output_dir (str): The directory to save the model file.
    """
    try:
        if model is None:
            raise ValueError("No model available to save")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_param("output_dir", str(output_dir))
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        with open(output_dir / "xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved successfully to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving model to {output_dir if 'output_dir' in locals() else 'output directory'}: {e}")

def main():
    mlflow.set_tracking_uri("http://ec2-3-15-213-179.us-east-2.compute.amazonaws.com:5000/")
    mlflow.set_experiment("XGBoost")
    with mlflow.start_run():
        data = pd.read_csv("/Users/jatin/Desktop/MLOps/pipelines/feature_engineering/engineered_data.csv")
        X_train = data.drop('sentiment', axis=1)
        y_train = data['sentiment']
        model = train_model(X_train, y_train)
        save_model(model)
if __name__ == "__main__":
    main()