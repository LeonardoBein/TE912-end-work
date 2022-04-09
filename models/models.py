from dataclasses import dataclass
from typing import Any, Tuple
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class CNNModel1:
    """LeNet CNN Model"""

    # Model / data parameters
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    
    model: Any  = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)),
            layers.AveragePooling2D(),
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            layers.AveragePooling2D(),
            layers.Flatten(),
            layers.Dense(units=120, activation='relu'),
            layers.Dense(units=84, activation='relu'),
            layers.Dense(units=10, activation = 'softmax'),
        ]
    )

@dataclass
class CNNModel2:
    """AlexNet CNN Model"""

    # Model / data parameters
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    

    model: Any  = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            
            #1st Convolutional Layer
            layers.Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

            #2nd Convolutional Layer
            layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            
            #3rd Convolutional Layer
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            #4th Convolutional Layer
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            #5th Convolutional Layer
            layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
    
            #Passing it to a Fully Connected layer
            layers.Flatten(),
            
            # 1st Fully Connected Layer
            layers.Dense(4096, input_shape=(32,32,3,)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            # Add Dropout to prevent overfitting
            layers.Dropout(0.4),

            #2nd Fully Connected Layer
            layers.Dense(4096),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            #Add Dropout
            layers.Dropout(0.4),

            #3rd Fully Connected Layer
            layers.Dense(1000),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            #Add Dropout
            layers.Dropout(0.4),

            #Output Layer
            layers.Dense(10),
            layers.BatchNormalization(),
            layers.Activation('softmax'),
        ]
    )

@dataclass
class CNNModel3:
    """VGGNet CNN Model"""

    # Model / data parameters
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (28, 28, 1)

    model: Any  = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax'),
        ]
    )
    
@dataclass
class CNNModel4:
    """Docstring for CNNModel."""

    # Model / data parameters
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    

    model: Any  = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

@dataclass
class CNNModel5:
    """Docstring for CNNModel."""

    # Model / data parameters
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    

    model: Any  = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D((2, 2),padding='same'),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation='linear',padding='same'),
            layers.LeakyReLU(alpha=0.1),
            layers.MaxPooling2D(pool_size=(2, 2),padding='same'),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation='linear',padding='same'),
            layers.LeakyReLU(alpha=0.1),                  
            layers.MaxPooling2D(pool_size=(2, 2),padding='same'),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(128, activation='linear'),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax'),
        ]
    )