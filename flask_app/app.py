from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
from mlflow_creds import MLFLOW_TRACKING_URI, REPO_NAME, REPO_OWNER


import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore')

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def remove_numbers(text):
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    text = [word.lower() for word in text.split()]
    return " ".join(text)

def remove_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace(':', '')
    text = re.sub('\s+', ' ', text).strip()
    return text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    '''Remove sentences with less than 3 words'''
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = remove_numbers(text)
    text = remove_urls(text)
    text = remove_punctuations(text)
    text = remove_stop_words(text)
    text = lower_case(text)
    text = lemmatization(text)

    return text

# For local use
# ================================================================================================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
dagshub.init(repo_owner= REPO_OWNER, repo_name=REPO_NAME, mlflow= True)

# For Production use
# ================================================================================================
## Set up Dagshub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError('CAPSTONE_TEST environment variable not set')

# os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
# os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# dagshub_url = 'https://dagshub.com'
# repo_owner = REPO_OWNER
# repo_name = REPO_NAME

# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------------------

app = Flask(__name__)

# create a custom registry
registry = CollectorRegistry()

# custom metrics using the registry
REQUEST_COUNT = Counter('app_request_count', 'Total number of requests to the app', ['method', 'endpoint'], registry= registry)
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Latency of requests in seconds', ['endpoint'], registry= registry)
PREDICTION_COUNT = Counter('model_prediction_count', 'Count of predictions of each class', ['prediction'], registry= registry)

# Get the latest model version from MLflow reigstry
model_name = 'my_model'
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages= ['Staging'])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages= ['None'])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f'Fetching the mdoel from {model_uri}')
model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method='GET', endpoint='/').inc()
    start_time = time.time()
    response = render_template('index.html', result = None)
    REQUEST_LATENCY.labels(endpoint='/').observe(time.time() - start_time)
    return response

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint='/predict').inc()
    start_time = time.time()

    text = request.form['text']
    text = normalize_text(text)

    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    result = model.predict(features_df)
    prediction = result[0]

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start_time)
    return render_template('index.html', result=prediction)

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(registry), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)