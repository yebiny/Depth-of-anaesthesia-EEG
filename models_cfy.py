import os, sys, glob
import numpy as np
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

def cfy1(xshape, n_class, optimizer):
    
    model = models.Sequential()
    model.add(layers.Input(shape=xshape))
    
    for rate in (1,2,4,8):
        model.add(layers.Conv1D(filters=20, kernel_size=2, 
                                padding='causal', activation='relu', dilation_rate=rate))
    
    model.add(layers.AveragePooling1D(50, padding='same'))
    model.add(layers.Conv1D(20, 100, padding='same', activation='relu'))
    
    model.add(layers.Conv1D(1, 100, padding='same', activation='relu'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    
   
    if n_class==2:
        n_dim = n_class-1
        act = 'sigmoid'
        loss='binary_crossentropy'
    else:
        n_dim = n_class
        act = 'softmax'
        loss='sparse_categorical_crossentropy'
    
    # Last layer - for label
    model.add(layers.Conv1D(n_dim, 10, padding='same'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    model.add(layers.Reshape((n_dim, )))
    model.add(layers.Activation(act))
    
    model.compile(optimizer, loss, metrics=['accuracy'])    
    return model


def cfy2(xshape, n_class, optimizer):
    
    model = models.Sequential()
    model.add(layers.Input(shape=xshape))
    
    for rate in (1,2,4,8):
        model.add(layers.Conv1D(filters=20, kernel_size=2, 
                                padding='causal', activation='relu', dilation_rate=rate))
    
    model.add(layers.AveragePooling1D(50, padding='same'))
    model.add(layers.Conv1D(20, 100, padding='same', activation='relu'))
    
    model.add(layers.Conv1D(1, 100, padding='same', activation='relu'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    
    if n_class==2:
        n_dim = n_class-1
        act = 'sigmoid'
        loss='binary_crossentropy'
    else:
        n_dim = n_class
        act = 'softmax'
        loss='sparse_categorical_crossentropy'
    
    model.add(layers.Conv1D(n_dim*4, 10, padding='same'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    model.add(layers.Reshape((n_dim*4, )))
    
    model.add(layers.Dense(n_dim))
    model.add(layers.Activation(act))

    model.compile(optimizer, loss, metrics=['accuracy'])    
    return model

def dense1(xshape, n_class, optimizer):
    model = models.Sequential()
    model.add(layers.Input(shape=xshape))

    model.add(layers.Dense(n_dim))
    model.add(layers.Activation(act))

MODELS = {
    'cfy1': cfy1,
    'cfy2': cfy2,
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
