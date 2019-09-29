import logging
from data import DataPipeline
from model import ModelLSTM

def train(use_cuda: bool):
    data = DataPipeline()
    data.preprocess()
    logger.info('Finished preprocessing data for model training')
    model = ModelLSTM(use_cuda)
    model.train(data.train)
    f1 = model.assess_model_performance(data.test)
    logger.info('New model achieved f1 score of {} on test set'.format(f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, help='use GPU for training')
    args = parser.parse_args()
    train(args.cuda)