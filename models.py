import os, sys, glob
import numpy as np
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

def wavenet_class(xshape, optimizer):
    
    model = models.Sequential()
    model.add(layers.Input(shape=xshape))
    
    for rate in (1,2,4,8):
        model.add(layers.Conv1D(filters=20, kernel_size=2, 
                                padding='causal', activation='relu', dilation_rate=rate))
    
    model.add(layers.AveragePooling1D(50, padding='same'))
    model.add(layers.Conv1D(20, 100, padding='same', activation='relu'))
    
    model.add(layers.Conv1D(1, 100, padding='same', activation='relu'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    
   
    # Last layer - for label
    model.add(layers.Conv1D(4, 10, padding='same'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    model.add(layers.Reshape((4, )))
    model.add(layers.Activation('softmax'))

    loss='sparse_categorical_crossentropy'
    model.compile(optimizer, loss, metrics=['accuracy'])    
    
    return model

def wavenet_class_binary(xshape, optimizer):

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=[xshape[1], xshape[2]]))
    
    for rate in (1,2,4,8):
        model.add(layers.Conv1D(filters=20, kernel_size=2, 
                                padding='causal', activation='relu', dilation_rate=rate))
    
    model.add(layers.AveragePooling1D(50, padding='same'))
    model.add(layers.Conv1D(20, 100, padding='same', activation='relu'))
    
    model.add(layers.Conv1D(1, 100, padding='same', activation='relu'))
    model.add(layers.AveragePooling1D(10, padding='same'))
   
    # Last layer - for label
    model.add(layers.Conv1D(1, 10, padding='same'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    model.add(layers.Reshape((1, )))
    model.add(layers.Activation('sigmoid'))
    
    loss='binary_crossentropy'
    model.compile(optimizer, loss, metrics=['accuracy'])    
    return model    

def wavenet_regression(xshape, optimizer):
    x = layers.Input(shape=xshape)
    y = layers.Conv1D(filters=20, kernel_size=2, padding='causal', activation='relu', dilation_rate=1)(x)
    y = layers.Conv1D(filters=20, kernel_size=2, padding='causal', activation='relu', dilation_rate=2)(y)
    y = layers.Conv1D(filters=20, kernel_size=2, padding='causal', activation='relu', dilation_rate=4)(y)
    y = layers.Conv1D(filters=20, kernel_size=2, padding='causal', activation='relu', dilation_rate=8)(y)

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
    'class': wavenet_class,
    'class_bn': wavenet_class_binary,
    'regression': wavenet_regression,
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
