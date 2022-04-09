import numpy as np
from tensorflow import keras
from models import CNNModelAbstract
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train_model(model_template: CNNModelAbstract, model_number: int):
    
    # Model / data parameters
    num_classes = model_template.num_classes

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # split data: 80% (48000) for train model and 20% (12000) for valid model
    train_X,valid_X,train_label,valid_label = train_test_split(x_train, y_train, test_size=0.2, random_state=13)
    
    # train the model
    batch_size = 256
    epochs = 20

    model_template.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_train = model_template.model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, validation_data=(valid_X, valid_label))

    # save the model
    path = '/home/leonardo/Documentos/UFPR/ia/te941-work-2/model_train'
    try:
        model_template.model.save(path + '/mnist_model_' + str(model_number) + '.h5')
        print('Model ' + str(model_number) + ' Saved!')
        model_train.model.save_weights(path + '/mnist_model_weights_' + str(model_number) + '.h5')
        print('Model Weights ' + str(model_number) + ' Saved!')
        
    except Exception as error:
        print(error)
