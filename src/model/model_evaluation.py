import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.logger import logging
import mlflow
import mlflow.sklearn
import dagshub
import os


# Dagshub/MLflow for production
# ===========================================================================
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError(f'CAPSTONE_TEST environment variable not set')

# os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
# os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# dagshub_uri = "https://dagshub.com"
# repo_owner = "owner"
# repo_name = "repo"

# mlflow.set_tracking_uri(f'{dagshub_uri}/{repo_owner}/{repo_name}.mlflow')

# Dagshub/MLflow for local
# ============================================================================
mlflow.set_registry_uri('https://dagshub.com/kirksalvator6392/capstone_project.mlflow')
dagshub.init(repo_owner='kirksalvator6392', repo_name='capstone_project', mlflow=True)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f'Model loaded from {file_path}')
        return model
    except FileNotFoundError:
        logging.error(f'File not found at {file_path}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occured while loading the model for evaluation: {e}')
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded from {file_path}')
        return df
    except pd.errors.ParserError as e:
        logging.error(f'Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occured while loading the data for model evaluation: {e}')
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info(f'Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error(f'Error during model evaluation: {e}')
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent= 4)
        logging.info(f'Model info saved at {file_path}')
    except Exception as e:
        logging.error(f'Unexpected error occured while saving model info: {e}')
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent= 4)
        logging.info(f'Metrics saved to {file_path}')
    except Exception as e:
        logging.error(f'Unexcepted error occured while saving metrics: {e}')
        raise

def main():
    mlflow.set_experiment('my_dvc_pipeline')
    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:,:-1].values
            y_test = np.array(test_data.iloc[:, -1].values)

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')

            for metric_name, metric_val in metrics.items():
                mlflow.log_metric(metric_name, metric_val)

            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_val in params.items():
                    mlflow.log_param(param_name, param_val)
            mlflow.sklearn.log_model(clf, 'model')

            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
        
        except Exception as e:
            logging.error(f'Failed to complete the model evaluation. Error: {e}')

if __name__ == '__main__':
    main()