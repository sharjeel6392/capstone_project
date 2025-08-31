import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
# from src.connections import s3_connections

def load_params(params_path: str) -> dict:
    """
        Load parameters from a YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug(f'Parameters retrieved from {params_path}')
        return params
    except FileNotFoundError:
        logging.error(f'File not found {params_path}')
        raise
    except yaml.YAMLError as e:
        logging.error(f'YAML error: {e}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error: {e}')
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """
        Load data from a csv file
    """
    try:
        df = pd.read_csv(data_url)
        logging.info(f'Data loaded from {data_url}')
        print(df.head())
        return df
    except pd.errors.ParserError as e:
        logging.error(f'Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
        Preprocess the data
    """
    try:
        logging.info('Pre-processing...')
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        logging.info('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logging.error(f'Missing column in the dataframe: {e}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occured while preprocessing: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
        Split the data into train and test set and save them separately
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok= True)
        print(f'filepath: {raw_data_path}')
        train_data.to_csv(os.path.join(raw_data_path, 'train_data.csv'), index= False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'), index= False)
        logging.debug(f'Train and Test data saved to {raw_data_path}')
    except Exception as e:
        logging.error(f'Unexpected error while saving the data: {e}')
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2

        df = load_data(data_url= './notebooks/data.csv')
        # s3 = s3_connections.s3_operations('bucket-name', 'access-key', 'secret-key')
        # df = s3.fetch_file_from_s3('data.csv')

        if df is None:
            logging.error('Loaded dataframe is None. Aborting preprocessing.')
            raise ValueError('Loaded dataframe is None.')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    
    except Exception as e:
        logging.error(f'Failed to complete the data ingestion process: {e}')
        raise

if __name__ == '__main__':
    main()