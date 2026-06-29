FROM python:3.8-slim

WORKDIR /app
RUN apt-get update && apt-get install -y git  # DVC needs git

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt dvc[s3] \
    && python -m nltk.downloader punkt stopwords wordnet

COPY . .

# Pull the model file from S3 during build
# You must provide AWS credentials as build-args or via IAM role
RUN dvc pull pipelines/model/xgboost_model.pkl

CMD ["python", "app.py"]