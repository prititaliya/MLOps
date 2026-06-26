import pandas as pd
import numpy as np
import logging
from pathlib import Path
import mlflow
import os
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"

# Prevent MLflow from attempting complex thread pool socket connection validation checks
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
mlflow.set_tracking_uri("http://ec2-3-15-213-179.us-east-2.compute.amazonaws.com:5000/")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def load_data(file_path="/Users/jatin/Desktop/MLOps/data/archive/twitter_training.csv"):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        mlflow.log_param("file_path", file_path)
        data = pd.read_csv(file_path,header=None)
        data.columns = ['id', 'entity', 'sentiment', 'tweet']
        logging.info(f"Data loaded successfully from {file_path}")

        data.dropna(inplace=True)
        data.drop('id', axis=1, inplace=True)
        samped_data = data.sample(frac=0.2, random_state=42)
        return samped_data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def save_data(data, output_dir="/Users/jatin/Desktop/MLOps/pipelines/data_ingestion"):
    """
    Save the processed data to a CSV file.

    Parameters:
    data (pd.DataFrame): The processed data to save.
    output_dir (str): The directory to save the CSV files.
    """
    try:
        if data is None:
            raise ValueError("No data available to save")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_data = data.sample(frac=0.8, random_state=42)
        test_data = data.drop(train_data.index)
        train_data.to_csv(output_dir / "train_data.csv", index=False)
        test_data.to_csv(output_dir / "test_data.csv", index=False)
        logging.info(f"Data saved successfully to {output_dir}")
        mlflow.log_artifact(str(output_dir / "train_data.csv"))
    except Exception as e:
        logging.error(f"Error saving data to {output_dir if 'output_dir' in locals() else 'output directory'}: {e}")
def main():
    mlflow.set_experiment("XGBoost")
    with mlflow.start_run():
        data = load_data()
        if data is not None:
            save_data(data)
            logging.info("First few rows of the data:")
            logging.info(data.head())
if __name__ == "__main__":
    main()