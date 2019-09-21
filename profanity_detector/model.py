import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence

from sklearn.metrics import f1_score
from typing import Optional, Dict

class ModelLSTM:
    def __init__(self):
        self.epochs = 20
        self.max_features = 10000
        self.max_len = 100
        self.tokenizer: text.Tokenizer
        self.model: Optional[Sequential]

    def one_hot_encode(self, y: pd.Series) -> np.array:
        return np.eye(2)[np.array(y).reshape(-1)]

    def define_model(self) -> Sequential:
        model = Sequential()
        model.add(Embedding(self.max_features, self.embed_size))
        model.add(Bidirectional(LSTM(250, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def fit_tokenizer(self, X: pd.Series) -> np.array:
        self.tokenizer = text.Tokenizer(num_words=self.max_features)
        X_tokenized = self.tokenizer.fit_on_texts(X)
        X_padded = sequence.pad_sequences(X_tokenized, maxlen=self.max_len)
        return X_padded

    def train(self, data: pd.DataFrame):
        X = data['comment_text']
        y = data['toxic']
        y = self.one_hot_encode(y)
        X = self.fit_tokenizer(X)

        saved_model_path = os.path.join(os.getcwd(), 'saved_model/1')

        self.model = self.define_model()
        self.model.fit(X, y)      
        self.model.export_savedmodel(saved_model_path)

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