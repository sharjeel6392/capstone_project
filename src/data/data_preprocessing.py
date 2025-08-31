import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from src.logger import logging
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_dataframe(df: pd.DataFrame, col: str = 'text') -> pd.DataFrame:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text: str) -> str:
        text = re.sub(r'https?://\S+ | www\.\S+', '', text)
        text = ''.join([char for char in text if not char.isdigit()])
        text = text.lower()
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace(':', '')
        text = re.sub('\s+', ' ', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    df[col] = df[col].apply(preprocess_text)

    df = df.dropna(subset=[col])
    logging.info('Data preprocessing completed.')
    return df

def main():
    try:
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')
        logging.info(f'Train and test data loaded for preprocessing.')

        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')

        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok= True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index = False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index = False)

        logging.info(f'Processed data saved to {data_path}')
    except Exception as e:
        logging.error(f'Data preprocessing failed due to: {e}')

if __name__ == '__main__':
    main()
        
        
