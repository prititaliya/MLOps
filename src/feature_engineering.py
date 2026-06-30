import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import mlflow
from src.paths import project_path
mlflow.set_tracking_uri("http://ec2-3-15-213-179.us-east-2.compute.amazonaws.com:5000/")

def feature_engineering(data):
    """Perform feature engineering on the preprocessed data.
    Parameters:
    data (pd.DataFrame): The preprocessed data to perform feature engineering on.
    Returns:
    pd.DataFrame: The data with engineered features.
    """
    try:    
        logging.info("Starting feature engineering")
        # Create a new feature for the number of words in the tweet
        data['tweet'] = data['tweet'].fillna('')
        data['num_words'] = data['tweet'].apply(lambda x: len(x.split()))

        # Create a new feature for the average word length in the tweet
        data['avg_word_length'] = data['tweet'].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)

        mlflow.log_param("feature_engineering_steps", "created num_words and avg_word_length features")
        return data
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        return None

def vectorize_data(data):
    """Vectorize the tweet data using TF-IDF.
    Parameters:
    data (pd.DataFrame): The data to vectorize. Must contain a 'tweet' column.  
    Returns:    
    pd.DataFrame: The vectorized data.
    """ 
    try:
        logging.info("Starting data vectorization")
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['tweet'])
        vectorized_data = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        vectorized_data['sentiment'] = data['sentiment']
        mlflow.log_param("vectorization_method", "CountVectorizer")
        return vectorized_data
    except Exception as e:
        logging.error(f"Error during data vectorization: {e}")
        return None

def save_engineered_data(data, output_dir="/Users/jatin/Desktop/MLOps/pipelines/feature_engineering"):
    """Save the engineered data to a CSV file.
    Parameters:
    data (pd.DataFrame): The engineered data to save.
    output_dir (str): The directory to save the CSV file.
    """
    try:
        if data is None:
            raise ValueError("No data available to save")

        output_dir = project_path("pipelines", "feature_engineering")
        output_dir.mkdir(parents=True, exist_ok=True)

        data.to_csv(output_dir / "engineered_data.csv", index=False)
        mlflow.log_artifact(str(output_dir / "engineered_data.csv"))
        logging.info(f"Engineered data saved successfully to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving engineered data to {output_dir if 'output_dir' in locals() else 'output directory'}: {e}") 
def main():
    data = pd.read_csv(project_path("pipelines", "data_preprocessing", "preprocessed_data.csv"))
    print(data.head())
    mlflow.set_experiment("XGBoost")
    with mlflow.start_run():
        if data is not None:
            engineered_data = feature_engineering(data)
            vectorized_data = vectorize_data(engineered_data)
            logging.info("First few rows of the engineered and vectorized data:")
            logging.info(vectorized_data.head())
            save_engineered_data(vectorized_data)
if __name__ == "__main__":   
     main()    