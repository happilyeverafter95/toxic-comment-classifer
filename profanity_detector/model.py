import os
import re
import logging
import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from keras.models import Sequential
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPooling1D, Dropout, InputLayer
from keras.preprocessing import text, sequence

from sklearn.metrics import f1_score
from typing import Optional, Dict

from keras import backend as K

class ModelLSTM:
    def __init__(self):
        self.epochs = 20
        self.max_features = 10000
        self.max_len = 100
        self.tokenizer = text.Tokenizer(num_words=self.max_features)
        self.embed_size = 128
        self.batch_size = 128
        self.model_weights: np.array

    def one_hot_encode(self, y: pd.Series) -> np.array:
        return np.eye(2)[np.array(y).reshape(-1)]

    def fit_tokenizer(self, X: pd.Series) -> np.array:
        self.tokenizer.fit_on_texts(X)
        X_tokenized = self.tokenizer.texts_to_sequences(X)
        print('Tokenizer succesfully fitted')
        X_padded = sequence.pad_sequences(X_tokenized, maxlen=self.max_len)
        print('Tokenized text has been successfully padded')

        return X_padded

    def define_model(self) -> Sequential:
        model = Sequential()
        model.add(Embedding(self.max_features, self.embed_size))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def create_tensorflow_session(self):
        # define model meta data
        version = re.sub('[^0-9]', '', str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        export_path = os.path.join(os.getcwd(), 'saved_model/{}'.format(version))

        # define look up table for tokenizer
        tags = list(self.tokenizer.word_index.keys())
        tokens = list(self.tokenizer.word_index.values())

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        K.set_session(sess)
        K.set_learning_phase(0)

        lookup_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(
            keys=tf.constant(tags), values=tf.constant(tokens)), 0)

        x_input = tf.placeholder(tf.string)
        x_input_processed = tf.string_split([x_input], delimiter=' ').values
        tokenized_input = lookup_table.lookup(x_input_processed)
        tokenized_input = tf.expand_dims(tf.reshape(tokenized_input, [-1]), 0)

        padded_input = tf.pad(tokenized_input, tf.constant([[0, 0], [0, self.max_len]]))
        padded_input = tf.slice(padded_input, [0, 0], [-1, self.max_len])

        # model re-definition 
        serving_model = Sequential()
        serving_model.add(InputLayer(
            input_tensor=padded_input, input_shape=(None, self.max_len)))
        serving_model.add(Embedding(self.max_features, self.embed_size))
        serving_model.add(Bidirectional(LSTM(
            100, return_sequences=True, dropout=0.1, recurrent_dropout=0.2)))
        serving_model.add(GlobalMaxPooling1D())
        serving_model.add(Dense(2, activation='softmax'))
        serving_model.compile(loss='categorical_crossentropy',
                                     optimizer='adam', metrics=['accuracy'])

        serving_model.set_weights(self.model_weights)

        x_info = tf.saved_model.utils.build_tensor_info(x_input)
        y_info = tf.saved_model.utils.build_tensor_info(serving_model.output)

        model_version_tensor = tf.saved_model.utils.build_tensor_info(tf.constant(version))

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': x_info},
            outputs={'prediction': y_info,
                     'model_version': model_version_tensor},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        init_op = tf.group(tf.local_variables_initializer())
        sess.run(init_op)

        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()

        logging.info('New model saved to {}'.format(export_path))

    def train(self, data: pd.DataFrame):
        X = data['comment_text']
        y = data['toxic']
        y = self.one_hot_encode(y)
        X = self.fit_tokenizer(X)

        self.model = self.define_model()
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)
        self.model_weights = self.model.get_weights()

        self.create_tensorflow_session()

    def assess_model_performance(self, data: pd.DataFrame) -> float:
        assert self.model is not None, 'ERROR: no model trained'
        X = data['comment_text']
        y = data['toxic']
        X_tokenized = self.tokenizer.texts_to_sequences(X)
        X_padded = sequence.pad_sequences(X_tokenized, maxlen=self.max_len)
        y = self.one_hot_encode(y)
        y_pred = self.model.predict(X_padded)
        return f1_score(y_pred, y)