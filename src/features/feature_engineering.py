import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
import pickle
import constants

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f'Parameters loaded form {params_path}')
        return params
    except FileNotFoundError:
        logging.error(f'File {params_path} not found')
        raise
    except yaml.YAMLError as e:
        logging.error(f'Error reading YAML file: {e}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occured: {e}')
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info(f'Data loaded from {file_path} and NaNs are filled in')
        return df
    except pd.errors.ParserError as e:
        logging.error(f'Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occured while loading data for feature engineering: {e}')
        raise

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        logging.info(f'Applying Bag of Words (BoW)')
        vectorizer = CountVectorizer(max_features= max_features)

        X_train =  train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        os.makedirs(constants.MODEL_DIR, exist_ok=True)
        vectorizer_file_path = os.path.join(constants.MODEL_DIR, constants.VECTORIZER_FILE)

        pickle.dump(vectorizer, open(vectorizer_file_path, 'wb'))
        logging.info('Bag of words applied and data transformed!')

        return train_df, test_df
    except Exception as e:
        logging.error(f'Unexpected error occured while applying feature engineering: {e}')
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        df.to_csv(file_path, index = False)
        logging.info(f'Transformed data saved to {file_path}')
    except Exception as e:
        logging.error(f'Unexpected error occured while saving data: {e}')
        raise

def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        # max_features = int(20)

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_bow(train_data, test_data, max_features)
        save_data(train_df, os.path.join('./data', 'processed', 'train_bow.csv'))
        save_data(test_df, os.path.join('./data', 'processed', 'test_bow.csv'))


    except Exception as e:
        logging.error(f'Failed to complete feature engineering processes: {e}')
        raise

if __name__ == '__main__':
    main()