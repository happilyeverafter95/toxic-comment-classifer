import logging
from data import DataPipeline
from model import ModelLSTM

def train():
    data = DataPipeline()
    data.preprocess()
    logging.info('Finished preprocessing data for model training')
    model = ModelLSTM()
    model.train(data.train)
    logging.info('New model completed trained')

if __name__ == "__main__":
    train()