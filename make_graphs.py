import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import pandas as pd
import tensorflow as tf
import numpy as np

num_classes = 10
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def accuracy_model (model_number):
    path = os.getcwd() + '/model_train'
    model_name = '/mnist_model_'
    csv_name = '/mnist_model_dataframe_'
    extension = '.h5'

    model = load_model(path + model_name + str(model_number) + extension)
    score = model.evaluate(x_test, y_test, verbose=0)
    df = pd.read_csv(path + csv_name + str(model_number) + '.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)

    epochs = range(len(df['accuracy']))
    _, ax = plt.subplots(2, 1)
    
    ax[0].plot(epochs, df['accuracy'], 'bo', label='Training accuracy')
    ax[0].plot(epochs, df['val_accuracy'], 'red', label='Validation accuracy')
    ax[0].title.set_text('Model' + ': {}'.format(model_number) + ' - Accuracy: {:.3f}'.format(score[1]))
    ax[0].legend()

    ax[1].plot(epochs, df['loss'], 'bo', label='Training loss')
    ax[1].plot(epochs, df['val_loss'], 'red', label='Validation loss')
    #ax[1].title.set_text('Model' + ': {}'.format(model_number) + ' - Accuracy: {:.3f}'.format(score[1]))
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # some correct number images
    predicted_classes = model.predict(x_test)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    test_y = np.argmax(y_test, axis=1)
    correct = np.where(predicted_classes == test_y)[0]
    
    print("Found %d correct labels" % len(correct))
    correct_print = len(correct)
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Correct {}".format(predicted_classes[correct], test_y[correct]))
        plt.tight_layout()
    plt.suptitle("Found " + str(correct_print) + " correct labels")
    plt.show()

    # some incorrect numbers images
    predicted_classes = model.predict(x_test)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    test_y = np.argmax(y_test, axis=1)
    correct = np.where(predicted_classes != test_y)[0]
    
    print("Found %d incorrect labels" % len(correct))
    incorrect_print = len(correct)
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Correct {}".format(predicted_classes[correct], test_y[correct]))
        plt.tight_layout()

    plt.suptitle("Found " + str(incorrect_print) + " incorrect labels")
    plt.show()

def compare_models():
    path_to_dir = os.getcwd() + '/model_train'
    suffix = '.h5'
    filenames = os.listdir(path_to_dir)
    list_models_name = [ filename for filename in filenames if filename.endswith( suffix ) ]

    list_models = list()
    for list_model_name in list_models_name:
        #path = os.getcwd() + 
        print(list_model_name)
        model = load_model(os.getcwd() + list_model_name)
        list_models.append(model)

