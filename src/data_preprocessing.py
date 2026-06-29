import pandas as pd
from sklearn.preprocessing import LabelEncoder  
import logging
import numpy as np
import nltk
import os 
from pathlib import Path
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import joblib
import mlflow   
from src.paths import project_path
mlflow.set_tracking_uri("http://ec2-3-15-213-179.us-east-2.compute.amazonaws.com:5000/")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def preprocess_data(data,predict=False):
    """
    Preprocess the data by encoding categorical variables.

    Parameters:
    data (pd.DataFrame): The input data to preprocess.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    try:
        logging.info("Starting data preprocessing")
        # Encode the 'sentiment' column using LabelEncoder
        label_encoder = LabelEncoder()
        if not predict:
            data['sentiment'] = label_encoder.fit_transform(data['sentiment'])
            joblib.dump(label_encoder, project_path("pipelines", "model", "label_encoder.pkl"))
            

        # remove urls from the tweets
        data['tweet'] = data['tweet'].str.replace(r'http\S+', '', regex=True)
        # lowercase the tweets
        data['tweet'] = data['tweet'].str.lower()

        # tokenize the tweets
        data['tweet'] = data['tweet'].apply(nltk.word_tokenize)

        # remove stop words from the tweets
        stop_words = set(nltk.corpus.stopwords.words('english'))
        data['tweet'] = data['tweet'].apply(lambda x: [word for word in x if word not in stop_words])

        # join the tokens back into a string
        data['tweet'] = data['tweet'].apply(lambda x: ' '.join(x))

        # remove punctuation from the tweets
        data['tweet'] = data['tweet'].str.replace(r'[^\w\s]', '', regex=True)

        # remove numbers from the tweets
        data['tweet'] = data['tweet'].str.replace(r'\d+', '', regex=True)

        # remove extra whitespace from the tweets
        data['tweet'] = data['tweet'].str.strip()

        #remove mentions from the tweets
        data['tweet'] = data['tweet'].str.replace(r'@\w+', '', regex=True)

        #remove hashtags from the tweets
        data['tweet'] = data['tweet'].str.replace(r'#\w+', '', regex=True)

        #remove special characters from the tweets
        data['tweet'] = data['tweet'].str.replace(r'[^\w\s]', '', regex=True)   

        mlflow.log_param("preprocessing_steps", "removed URLs, lowercase, tokenized, removed stop words, joined tokens, removed punctuation, removed numbers, stripped whitespace, removed mentions, removed hashtags, removed special characters")
        return data
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return None

def lemmatize_data(data):
    """
    Lemmatize the tweets in the data.

    Parameters:
    data (pd.DataFrame): The input data to lemmatize.

    Returns:
    pd.DataFrame: The lemmatized data.
    """
    try:
        logging.info("Starting lemmatization")
        lemmatizer = nltk.stem.WordNetLemmatizer()
        data['tweet'] = data['tweet'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
        mlflow.log_param("lemmatization", "applied WordNet lemmatizer")
        return data
    except Exception as e:
        logging.error(f"Error during lemmatization: {e}")
        return None
def load_data(file_path="/Users/jatin/Desktop/MLOps/pipelines/data_ingestion/train_data.csv"):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        file_path = project_path("pipelines", "data_ingestion", "train_data.csv")
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def save_data(data, output_dir="/Users/jatin/Desktop/MLOps/pipelines/data_preprocessing"):
    """
    Save the processed data to a CSV file.

    Parameters:
    data (pd.DataFrame): The processed data to save.
    output_dir (str): The directory to save the CSV files.
    """
    try:
        if data is None:
            raise ValueError("No data available to save")

        output_dir = project_path("pipelines", "data_preprocessing")
        output_dir.mkdir(parents=True, exist_ok=True)

        data.to_csv(output_dir / "preprocessed_data.csv", index=False)
        mlflow.log_artifact(str(output_dir / "preprocessed_data.csv"))
        logging.info(f"Data saved successfully to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving data to {output_dir if 'output_dir' in locals() else 'output directory'}: {e}")
def main():
    data = load_data()
    mlflow.set_experiment("XGBoost")
    with mlflow.start_run():
        if data is not None:
            preprocessed_data = preprocess_data(data)
            if preprocessed_data is not None:
                processed_data = lemmatize_data(preprocessed_data)
                if processed_data is not None:
                    print(processed_data.head())
                    save_data(processed_data)


if __name__ == "__main__":
    main()