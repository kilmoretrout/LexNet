import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate, BatchNormalization, AveragePooling1D
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, Conv1D, UpSampling2D

def MLP(input_shape = (2048,)):
    input = Input(input_shape)
    x = Dense(128)(input)
    x = Dense(128)(x)
    x = Dense(3, activation = 'softmax')(x)

    model = Model(input, x)

    return model

def Lex_classifier(input_shape = (1280, 64)):
    input = Input(input_shape)
    x = Conv1D(256, kernel_size = 2, activation = 'relu')(input)
    x = Conv1D(128, kernel_size = 2, activation = 'relu')(x)
    x = AveragePooling1D(pool_size = 2)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(128, kernel_size = 2, activation = 'relu')(x)
    x = AveragePooling1D(pool_size = 2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation = 'softmax')(x)

    model = Model(input, x)

    return model