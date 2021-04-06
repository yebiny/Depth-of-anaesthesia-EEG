import os, sys, glob
import numpy as np
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split


def reg1(xshape, optimizer):
    def _conv(x):
        y = layers.Conv1D( filters=20
                         , kernel_size=2
                         , padding='causal'
                         , activation='relu'
                         , dilation_rate=1)(x)
        return y
    
    x = layers.Input(shape=xshape)
    y = _conv(x)
    for rate in (2,4,8):
        y = _conv(y)

    y = layers.AveragePooling1D(50, padding='same')(y)
    y = layers.Conv1D(20, 100, padding='same', activation='relu')(y)

    y = layers.Conv1D(10, 100, padding='same', activation='relu')(y)
    y = layers.AveragePooling1D(100, padding='same')(y)
    
    # Last layer - for label
    y = layers.Conv1D(10, 10, padding='same')(y)
    y = layers.AveragePooling1D(10, padding='same')(y)
    y = layers.Reshape((10, ))(y)
    
    y = layers.Dense(10, activation='relu')(y)
    y = layers.Dense(1)(y) 
    
    model = models.Model(x, y)
    model.compile(optimizer, 'mse', metrics=['mse', 'mae'])

    return model


MODELS = {
    'reg1': reg1,
}


def main():

    dataDir = sys.argv[1]
    model_name = sys.argv[2]

    # load data
    x_train, x_valid, _, y_train, y_valid, _ = load_data(dataDir)
    
    # load and draw model
    model = MODELS[model_name](x_train, 'adam')
    model.summary()

if __name__=='__main__':
    main()
