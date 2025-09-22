import os
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.linear_model import LogisticRegression
import pickle
from src.constants import MODEL_DIR, MODEL_FILE

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded from {file_path}')
        return df
    except pd.errors.ParserError as e:
        logging.error(f'Failed to parse the csv file: {e}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occured while loading the data for model building: {e}')
        raise
        
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    try:
        clf = LogisticRegression(C = 1, solver = 'liblinear', penalty= 'l2')
        clf.fit(X_train, y_train)
        logging.info(f'Model trained successfully')
        return clf
    except Exception as e:
        logging.error(f'Unexpected error occured during training: {e}')
        raise

def save_model(model: LogisticRegression, file_path: str) -> None:
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f'Model saved to {file_path}')
    except Exception as e:
        logging.error(f'Unexpected error occured while saving the model: {e}')
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = np.array(train_data.iloc[:, -1])

        clf = train_model(X_train, y_train)
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_file_path = os.path.join(MODEL_DIR, MODEL_FILE)

        save_model(clf, model_file_path)
    except Exception as e:
        logging.error(f'Failed to complete the model building process: {e}')
    
if __name__ == '__main__':
    main()