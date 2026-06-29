FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt stopwords wordnet

COPY . .


CMD ["python", "app.py"]