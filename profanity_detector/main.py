import logging
from data import DataPipeline
from model import ModelLSTM

def train(use_cuda: bool):
    data = DataPipeline()
    data.preprocess()
    logging.info('Finished preprocessing data for model training')
    model = ModelLSTM()
    model.train(data.train)
    f1 = model.assess_model_performance(data.test)
    logging.info('New model achieved f1 score of {} on test set'.format(f1))