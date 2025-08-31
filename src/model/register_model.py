import json
import mlflow
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore')

# For production:
# ======================================================================================
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError(f'The environment variable CAPSTONE_TEST is not set.')

# os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
# os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# dagshub_url = 'https://dagshub.com'
# repo_owner = '<owner>'
# repo_name = '<repo>'

# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# For local:
# =======================================================================================
mlflow.set_tracking_uri('https://dagshub.com/owner/repo.mlflow')
dagshub.init(repo_owner='owner', repo_name='repo', mlflow=True)

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info(f'Model info loaded from {file_path}')
        return model_info
    except FileNotFoundError:
        logging.error(f'{file_path} not found!')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occured while loading model info: {e}')
        raise

def register_model(model_name: str, model_info: dict) -> None:
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage= 'Staging'
        )
        logging.info(f'Model {model_name} version {model_version.version} registered and transitioned into Staging')
    except Exception as e:
        logging.error(f'Unexpected error occured during model registry: {e}')
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
        logging.info(f'Model registerd!')
    except Exception as e:
        logging.error(f'Failed to complete model registration process. Error: {e}')

if __name__ == '__main__':
    main()