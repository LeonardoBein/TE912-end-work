import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import numpy as np
from pprint import pprint
import pandas as pd
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

pathHere = os.getcwd()
modeloTreinadoArquivo = f"{pathHere}/model_train"

inputShape = (28, 28, 1)
# Model / data parameters
numClasses = 10
cnnModeloUtilizado = keras.Sequential(
    [
        keras.Input(shape=inputShape),
        layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="linear",
            padding="same",
            input_shape=inputShape,
        ),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="linear", padding="same"),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation="linear", padding="same"),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(128, activation="linear"),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        layers.Dense(numClasses, activation="softmax"),
    ]
)

# Dataset: MNIST-like fashion product database
# 60000 images for datasetTrain and 10000 images for datesetTest
(xTrain, yTrain), (xTest, yTest) = keras.datasets.fashion_mnist.load_data()


datasetTrain = [xTrain, yTrain]
datesetTest = [xTest, yTest]

##### Normalização #####

# Scale images to the [0, 1] range
datasetTrain[0] = datasetTrain[0].astype("float32") / 255
datesetTest[0] = datesetTest[0].astype("float32") / 255

# Make sure images have shape (28, 28, 1)
datasetTrain[0] = np.expand_dims(datasetTrain[0], -1)
datesetTest[0] = np.expand_dims(datesetTest[0], -1)

# convert class vectors to binary class matrices
datasetTrain[1] = keras.utils.to_categorical(datasetTrain[1], numClasses)
datesetTest[1] = keras.utils.to_categorical(datesetTest[1], numClasses)

########################


def treinar(dataset):
    global inputShape, numClasses, cnnModeloUtilizado, modeloTreinadoArquivo

    if os.path.exists(f"{modeloTreinadoArquivo}.h5"):
        respo = input("Deseja treinar novamente ? [S/n] ")

        if respo == "n" or respo == "N":
            return

    # train the model
    cnnModeloUtilizado.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    print("Treinando modelo ...")

    modeloTreinado = cnnModeloUtilizado.fit(
        dataset[0], dataset[1], batch_size=256, epochs=10, validation_split=0.1
    )

    # Salvando historico de treinamento

    history = pd.DataFrame(
        {
            "accuracy": modeloTreinado.history["accuracy"],
            "val_accuracy": modeloTreinado.history["val_accuracy"],
            "loss": modeloTreinado.history["loss"],
            "val_loss": modeloTreinado.history["val_loss"],
        }
    )

    print("modelo treinado !")

    pprint(history)

    # salvar modelo para usar
    cnnModeloUtilizado.save(f"{modeloTreinadoArquivo}.h5")
    history.to_excel(f"{modeloTreinadoArquivo}.xlsx")
    history.to_csv(f"{modeloTreinadoArquivo}.csv")


def rodarTest(dataset):
    global modeloTreinadoArquivo
    modeloTreinado = load_model(f"{modeloTreinadoArquivo}.h5")
    score = modeloTreinado.evaluate(dataset[0], dataset[1], verbose=0)

    df = pd.read_csv(f"{modeloTreinadoArquivo}.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True)

    epochs = range(len(df["accuracy"]))
    _, ax = plt.subplots(2, 1)

    ax[0].plot(epochs, df["accuracy"], "bo", label="Training accuracy")
    ax[0].plot(epochs, df["val_accuracy"], "red", label="Validation accuracy")
    ax[0].title.set_text("Modelo  - Accuracy: {:.3f}".format(score[1]))
    ax[0].legend()

    ax[1].plot(epochs, df["loss"], "bo", label="Training loss")
    ax[1].plot(epochs, df["val_loss"], "red", label="Validation loss")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # some correct number images
    predictedClasses = modeloTreinado.predict(dataset[0])
    predictedClasses = np.argmax(np.round(predictedClasses), axis=1)
    test_y = np.argmax(dataset[1], axis=1)

    correct = np.where(predictedClasses == test_y)[0]
    incorrect = np.where(predictedClasses != test_y)[0]

    print("Total test %d" % len(test_y))
    print("Found %d correct labels" % len(correct))
    print("Found %d incorrect labels" % len(incorrect))


if __name__ == "__main__":

    treinar(datasetTrain)
    rodarTest(datesetTest)
