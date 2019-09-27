import os
import subprocess
import zipfile
import pandas as pd
import numpy as np
import kaggle 

from typing import Optional
from config import KAGGLE_USERNAME, KAGGLE_KEY

kaggle.api.authenticate()

os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

class DataPipeline:
    def __init__(self):
        self.preprocessing_steps = [self.lowercase,
                                    self.impute_missing_comments,]
        self.train: Optional[pd.DataFrame]
        self.test: Optional[pd.DataFrame]

    # Kaggle API is smart enough to know when a file has been downloaded
    def load_data(self):
        data_path = os.path.join(os.getcwd(), 'data')
        kaggle.api.competition_download_files('jigsaw-toxic-comment-classification-challenge', path=data_path)
        with zipfile.ZipFile(data_path + '/jigsaw-toxic-comment-classification-challenge.zip') as z:
            with z.open('train.csv') as f:
                self.train = pd.read_csv(f)
            with z.open('test.csv') as f:
                self.test = pd.read_csv(f)

    def preprocess(self):
        self.load_data()
        for func in self.preprocessing_steps:
            self.train['comment_text'] = self.train['comment_text'].apply(func)
            self.test['comment_text'] = self.test['comment_text'].apply(func)
            print('Finished applying preprocessing step {}'.format(func.__name__))

    '''
        Transformative steps for data cleaning/preprocessing
    '''

    def lowercase(self, text: str) -> str:
        return text.lower()

    def impute_missing_comments(self, text: str) -> str:
        if text == '' or text is np.nan:
            return 'unknown'
        else:
            return text
    