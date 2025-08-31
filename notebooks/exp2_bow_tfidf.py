import setuptools
import scipy
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

# Configuration for Dagshub and MLflow
CONFIG = {
    'data_path': 'notebooks/data.csv',
    'test_size': 0.2,
    'mlflow_tracking_uri': '<link to your dagshub project/repo>',
    'dagshub_repo_owner': '<dagshub repo owner>',
    'dagshub_repo_name': '<dagshub repo name>',
    'mlflow_experiment_name': 'bag_of_words vs tfidf'
}

# Initialize Dagshub and MLflow
mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
dagshub.init(repo_owner = CONFIG['dagshub_repo_owner'], repo_name = CONFIG['dagshub_repo_name'], mlflow = True)
mlflow.set_experiment(CONFIG['mlflow_experiment_name'])


# Preprocessing functions
# 1. Lemmatization function
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

# 2. Stop words removal function
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    return text

# 3. Remove numbers
def remove_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

# 4. Convert to lowercase
def to_lowercase(text):
    return " ".join(word.lower() for word in text.split())

# 5. Remove punctuations
def remove_punctuation(text):
    text = re.sub('[%s]' %re.escape(string.punctuation), ' ', text)
    text = text.replace(';', ' ')
    text = re.sub('\s+', ' ', text)
    return text

# 6. Remove URLs
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].apply(to_lowercase)
        df['review'] = df['review'].apply(remove_stopwords)
        df['review'] = df['review'].apply(remove_numbers)
        df['review'] = df['review'].apply(remove_punctuation)
        df['review'] = df['review'].apply(remove_urls)
        df['review'] = df['review'].apply(lemmatize_text)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

# Models and Vectorizers to evaluate
VECTORIZERS = {
    'CountVectorizer': CountVectorizer(),
    'TfidfVectorizer': TfidfVectorizer()
}

MODELS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'RandomForestClassifier': RandomForestClassifier(),
    'XGBClassifier': XGBClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}


def train_and_evaluate_models(df):
    with mlflow.start_run(run_name = 'All_experiments') as parent_run:
        for model_name, model in MODELS.items():
            for vectorizer_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name = f'{model_name} Model with {vectorizer_name} Vectorizer', nested= True) as child_run:
                    try:
                        # Vectorization
                        X = vectorizer.fit_transform(df['review'])
                        y = df['sentiment']
                        X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=CONFIG['test_size'], random_state=42)

                        # Log processing parameters
                        mlflow.log_param('Vectorizer_params', vectorizer.get_params())
                        # mlflow.log_param({
                        #     'Vectorizer': vectorizer_name,
                        #     'Model': model_name,
                        #     'Test_size': CONFIG['test_size']
                        # })

                        # Train Model
                        curr_model = model
                        curr_model.fit(X_Train, y_Train)

                        # Log model parameters
                        log_model_params(model_name, curr_model)

                        # evaluate the model
                        y_Pred = curr_model.predict(X_Test)
                        metrics = {
                            'accruracy': accuracy_score(y_Test, y_Pred),
                            'precision': precision_score(y_Test, y_Pred),
                            'recall': recall_score(y_Test, y_Pred),
                            'f1_score': f1_score(y_Test, y_Pred)
                        }

                        mlflow.log_metrics(metrics)

                        # Log model
                        input_example = X_Test[:5] if not scipy.sparse.issparse(X_Test) else X_Test[:5].toarray()
                        mlflow.sklearn.log_model(curr_model, 'model', input_example = input_example)

                        # Print results for verification
                        print(f'Model: {model_name} with Vectorizer {vectorizer_name}:')
                        print(f'Metrics:\n{metrics}\n')

                    except Exception as e:
                        print(f'Error during training {model_name} with {vectorizer_name}: {e}')
                        mlflow.log_param('Error', str(e))
def log_model_params(model_name, model):
    """log model's hyperparameters to MLflow"""
    params_to_log = {}
    if model_name == 'LogisticRegression':
        params_to_log = {
            'C': model.C,
            'max_iter': model.max_iter,
            'solver': model.solver
        }
    elif model_name == 'MultinomialNB':
        params_to_log = {
            'alpha': model.alpha,
            'fit_prior': model.fit_prior
        }
    elif model_name == 'XGBClassifier':
        params_to_log = {
            'Learning_rate': model.learning_rate,
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
        }
    elif model_name == 'RandomForestClassifier':
        params_to_log = {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
        }
    elif model_name == 'GradientBoostingClassifier':
        params_to_log = {
            'learning_rate': model.learning_rate,
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth
        }

    mlflow.log_params(params_to_log)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f'Error loading data: {e}')
        raise

if __name__ == '__main__':
    df = load_data(CONFIG['data_path'])
    train_and_evaluate_models(df)