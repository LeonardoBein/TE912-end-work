import numpy as np
from tensorflow import keras
from models import CNNModelAbstract
import matplotlib.pyplot as plt
import os
import pandas as pd


def create_dataframe (model_train) -> pd.DataFrame:
    accuracy = model_train.history['accuracy']
    val_accuracy = model_train.history['val_accuracy']
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']

    data = {
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
        'loss': loss,
        'val_loss': val_loss
    }

    df = pd.DataFrame(data)

    return df


def train_model(model_template: CNNModelAbstract, model_number: int):
    
    # Model / data parameters
    num_classes = model_template.num_classes

    # the data, split between train and test sets
    # 60000 images for train and 10000 images for test
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
    
    # train the model
    batch_size = 256
    epochs = 20
    model_template.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print('Training model ' + str(model_number) + '...')
    model_train = model_template.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    df = create_dataframe(model_train)

    print('Model ' + str(model_number) + ' trained!')

    # save the model
    path = os.getcwd() + '/model_train'
    try:
        model_template.model.save(path + '/mnist_model_' + str(model_number) + '.h5')
        print('Model ' + str(model_number) + ' Saved!')
        df.to_csv(path + '/mnist_model_dataframe_' + str(model_number) + '.csv')
        print('Model Dataframe' + str(model_number) + ' Saved!')
        
    except Exception as error:
        print(error)