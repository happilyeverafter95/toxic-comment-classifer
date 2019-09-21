import os
import subprocess
import zipfile
import pandas as pd
import numpy as np

from config import KAGGLE_USERNAME, KAGGLE_KEY

os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

import kaggle

class DataPipeline:
    def __init__(self):
        self.preprocessing_steps = [self.lowercase,
                                    self.impute_missing_comments,]
        self.train: pd.DataFrame
        self.test: pd.DataFrame

    def load_data(self):
        data_path = os.path.join(os.getcwd(), 'data')
        cmd = ['kaggle', 'competitions', 'download', '-c', 'jigsaw-toxic-comment-classification-challenge', '-p', data_path]
        subprocess.Popen(cmd, stdout=subprocess.PIPE)
        with zipfile.ZipFile(data_path + '/jigsaw-toxic-comment-classification-challenge.zip') as z:
            with z.open('train.csv') as f:
                self.train = pd.read_csv(f)
            with z.open('test.csv') as f:
                self.test = pd.read_csv(f)
    
    def preprocess(self):
        if self.train is None:
            self.load_data()
        for func in self.preprocessing_steps:
            self.train['comment_text'] = self.train['comment_text'].apply(func)
            self.test['comment_text'] = self.test['comment_text'].apply(func)

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
    