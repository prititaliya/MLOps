import pandas as pd
import numpy as np
import logging
from pathlib import Path
from xgboost import XGBClassifier
import pickle
import joblib
from collections import Counter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# import mlflow
import nltk

from sklearn.preprocessing import LabelEncoder  

class ModelPredictor:
    def __init__(self, model_path="/pipelines/model/xgboost_model.pkl"):
        """Initialize the ModelPredictor with the path to the trained model.
        Parameters:
        model_path (str): The path to the trained model file.
        """
        self.model_path = Path(model_path)
        self.model = self.load_model()

    def load_model(self):
        """Load the trained model from a file.
        Returns:
        XGBClassifier: The loaded XGBoost model.
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            logging.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model from {self.model_path}: {e}")
            return None

    def predict(self, X):
        """Make predictions using the loaded model.
        Parameters:
        X (pd.DataFrame): The input features for prediction.
        Returns:
        np.ndarray: The predicted labels.
        """
        try:
            if self.model is None:
                raise ValueError("Model is not loaded. Cannot make predictions.")
            predictions = self.model.predict(X)
            logging.info("Predictions made successfully")
            return predictions
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            return None
    def preprocess_input(self, input_data):
        """
        Preprocess the data by encoding categorical variables.

        Parameters:
        data (pd.DataFrame): The input data to preprocess.

        Returns:
        pd.DataFrame: The preprocessed data.
        """
        try:
            logging.info("Starting data preprocessing")

            if input_data is None:
                raise ValueError("Input data is None.")
            if not isinstance(input_data, pd.DataFrame):
                input_data = pd.DataFrame(input_data)
            input_data = input_data.copy()

            if 'tweet' not in input_data.columns:
                raise ValueError("Input data must contain a 'tweet' column.")

            # Normalize tweet payloads from API requests (e.g., list tokens, numbers, nulls).
            input_data['tweet'] = input_data['tweet'].apply(
                lambda value: ' '.join(map(str, value)) if isinstance(value, list) else value
            )
            input_data['tweet'] = input_data['tweet'].fillna('').astype(str)

            if self.model is None:
                raise ValueError("Model is not loaded. Cannot preprocess input for prediction.")

            feature_names = getattr(self.model, "feature_names_in_", None)
            if feature_names is None:
                raise ValueError("Loaded model does not expose feature names for inference.")

            # remove urls from the tweets
            input_data['tweet'] = input_data['tweet'].str.replace(r'http\S+', '', regex=True)
            # lowercase the tweets
            input_data['tweet'] = input_data['tweet'].str.lower()

            # tokenize the tweets
            input_data['tweet'] = input_data['tweet'].apply(nltk.word_tokenize)

            # remove stop words from the tweets
            stop_words = set(nltk.corpus.stopwords.words('english'))
            input_data['tweet'] = input_data['tweet'].apply(lambda x: [word for word in x if word not in stop_words])

            # join the tokens back into a string
            input_data['tweet'] = input_data['tweet'].apply(lambda x: ' '.join(x))

            # remove punctuation from the tweets
            input_data['tweet'] = input_data['tweet'].str.replace(r'[^\w\s]', '', regex=True)

            # remove numbers from the tweets
            input_data['tweet'] = input_data['tweet'].str.replace(r'\d+', '', regex=True)

            # remove extra whitespace from the tweets
            input_data['tweet'] = input_data['tweet'].str.strip()

            # remove mentions from the tweets
            input_data['tweet'] = input_data['tweet'].str.replace(r'@\w+', '', regex=True)

            # remove hashtags from the tweets
            input_data['tweet'] = input_data['tweet'].str.replace(r'#\w+', '', regex=True)

            # remove special characters from the tweets
            input_data['tweet'] = input_data['tweet'].str.replace(r'[^\w\s]', '', regex=True)

            feature_rows = []
            for tweet in input_data['tweet'].fillna(''):
                token_counts = Counter(tweet.split())
                feature_rows.append({feature_name: token_counts.get(feature_name, 0) for feature_name in feature_names})

            return pd.DataFrame(feature_rows, columns=feature_names)
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            return None

def main():
    # mlflow.set_tracking_uri("http://ec2-3-15-213-179.us-east-2.compute.amazonaws.com:5000/")
    # mlflow.set_experiment("XGBoost")
    # with mlflow.start_run():
        # data = pd.read_csv("/Users/jatin/Desktop/MLOps/pipelines/data_preprocessing/preprocessed_data.csv")
    # X = data.drop('sentiment', axis=1)
    input_data = pd.DataFrame({
        'entity': ['CallOfDutyBlackopsColdWar'],
        'tweet': ["My name is prit"]
    })
    predictor = ModelPredictor()
    preprocessed_data = predictor.preprocess_input(input_data)
    if preprocessed_data is not None:
        predictions = predictor.predict(preprocessed_data)
        if predictions is not None:
            logging.info(f"Predictions: {predictions[:5]}")  # Log first 5 predictions
            encoder_path = Path("/pipelines/model/label_encoder.pkl")
            if encoder_path.exists():
                with open(encoder_path, "rb") as f:
                    label_encoder = joblib.load(f)
                decoded_predictions = label_encoder.inverse_transform(predictions)
                logging.info(f"Decoded Predictions: {decoded_predictions[:5]}")  # Log first 5 decoded predictions
            else:
                logging.error(f"Label encoder file not found at {encoder_path}. Cannot decode predictions.")
        else:
            logging.error("Prediction failed.")
    else:
        logging.error("Preprocessing failed. Cannot make predictions.")


if __name__ == "__main__":
    logging.info("Starting prediction process")
    main()
