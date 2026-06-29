FROM python:3.8-slim

WORKDIR /app

# Install git if needed, otherwise this is fine
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt punkt_tab stopwords wordnet

COPY . .

# Remove the 'RUN dvc pull...' line. 
# We will rely on the file being present in the repository.

CMD ["python", "app.py"]