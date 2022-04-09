import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
#from tensorflow.keras.models import load_weights

def accuracy_model (model_number):
    path = '/home/leonardo/Documentos/UFPR/ia/te941-work-2/model_train'
    model_name = '/mnist_model_'
    weights_name = '/mnist_model_weights_'
    extension = '.h5'

    model = load_model(path + model_name + str(model_number) + extension)
    model.load_weights('/home/leonardo/Documentos/UFPR/ia/te941-work-2/model_train/mnist_model_weights_1.h5')
    
def compare_models():
    pass