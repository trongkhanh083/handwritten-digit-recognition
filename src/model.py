import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

# Build LeNet-5 model
def lenet5_model(input_shape=(28, 28, 1)):
    inputs = Input(input_shape)

    x = Conv2D(6, (5, 5), activation='tanh')(inputs)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(16, (5, 5), activation='tanh')(x)
    x = AveragePooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(120, activation='tanh')(x)
    x = Dense(84, activation='tanh')(x)

    outputs = Dense(10, activation='softmax')(x)
    
    return Model(inputs, outputs)