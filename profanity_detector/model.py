import os
import re
import logging
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.python.saved_model import tag_constants

from sklearn.metrics import f1_score
from typing import Optional, Dict

class ModelLSTM:
    def __init__(self, use_cuda: bool):
        self.use_cuda: bool
        self.epochs = 20
        self.max_features = 10000
        self.max_len = 100
        self.tokenizer = text.Tokenizer(num_words=self.max_features)
        self.embed_size = 128
        self.batch_size = 128

    def one_hot_encode(self, y: pd.Series) -> np.array:
        return np.eye(2)[np.array(y).reshape(-1)]

    def fit_tokenizer(self, X: pd.Series) -> np.array:
        self.tokenizer.fit_on_texts(X)
        X_tokenized = self.tokenizer.texts_to_sequences(X)
        print('Tokenizer succesfully fitted')
        X_padded = sequence.pad_sequences(X_tokenized, maxlen=self.max_len)
        print('Tokenized text has been successfully padded')

        return X_padded

    def train(self, data: pd.DataFrame):
        X = data['comment_text']
        y = data['toxic']
        y = self.one_hot_encode(y)
        X = self.fit_tokenizer(X)

        version = re.sub('[^0-9]', '', str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        export_path = os.path.join(os.getcwd(), 'saved_model/'.format(version))

        model = Sequential()
        model.add(Embedding(self.max_features, self.embed_size))
        if self.use_cuda:
            model.add(Bidirectional(CuDNNLSTM(250, return_sequences=True)))
        else:
            model.add(Bidirectional(LSTM(250, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(2, activation='softmax'))
                
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

        tf.saved_model.simple_save(
            tf.keras.backend.get_session(),
            export_path,
            inputs={'input': model.input},
            outputs={'output': model.output})
        
        logging.info('New model saved to {}'.format(export_path))

    def assess_model_performance(self, data: pd.DataFrame) -> float:
        assert self.model is not None, 'ERROR: no model trained'
        X = data['comment_text']
        y = data['toxic']
        X_tokenized = self.tokenizer.texts_to_sequences(X)
        X_padded = sequence.pad_sequences(X_tokenized, maxlen=self.max_len)
        y = self.one_hot_encode(y)
        y_pred = self.model.predict(X_tokenized)
        return f1_score(pred, y)

    def predict(self, text: str) -> Dict[str, str]:
        assert self.model is not None, 'ERROR: no model trained'
        text = self.tokenizer.texts_to_sequences([text])
        text = sequence.pad_sequences(text, maxlen=self.max_len)
        prediction = self.model.predict(text)
        return {
            'prediction': 'toxic' if np.argmax(prediction) == 1 else 'not toxic'
        }