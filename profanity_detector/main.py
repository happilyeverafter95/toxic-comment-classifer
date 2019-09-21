import argparse

from data import DataPipeline
from model import ModelLSTM

from typing import Optional, Dict, Callable

model: Optional[ModelLSTM] = None

def train():
    data = DataPipeline().preprocess()
    global model
    model = ModelLSTM().train(data.train)
    f1 = model.assess_model_performance(data.test)
    print('Model sucessfully trained with test f1 '.format({f1}))


def predict(text: str) -> Dict[str, str]:
    if model is None:
        print('No model loaded into memory. Training new model.')
        train()
    return model.predict(text)

def no_function_found():
    raise ValueError('Not supported')

# TODO: support serve
def exe(arg: str) -> Callable:
    string_mapper = {'train': train, 'predict': predict}
    func = string_mapper.get(arg, no_function_found)
    return func()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description to go here')
    parser.add_argument('train', help='train model')
    parser.add_argument('predict', help='predict something')
    args = parser.parse_args()

    exe(args.name)
