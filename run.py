import argparse
from model_train.model_train import train_model
from models import Models
import graph
from os import system

def main (model_number: int, train_value: bool = False, compare_models: bool = False):
    
    modelName = f"MODEL_{model_number}"
    
    try:
        modelTrain = Models[modelName]

    except Exception as e:
        raise Exception(f"Model {model_number} not found")

    if train_value is True:
        train_model(modelTrain.value, model_number)
    else:
        if compare_models:
            graph.compare_models()
        else:
            # ler o arquivo treinado
            # plot acuracia do modelo
            graph.accuracy_model(model_number)
    
if __name__ == '__main__':
    system('clear')
    parser = argparse.ArgumentParser(description='Convolutional neural networks (CNNs)')
    parser.add_argument('model', metavar='N', type=int, help='Model number')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--compare', action=argparse.BooleanOptionalAction, default=False)             

    args = parser.parse_args()
    
    main(args.model, args.train, args.compare)