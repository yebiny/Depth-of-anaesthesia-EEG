import os, sys, glob
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_data(dataDir):
    x_data = np.load('%s/x_data.npy'%dataDir)
    y_data = np.load('%s/y_data.npy'%dataDir)
    x_test = np.load('%s/x_test.npy'%dataDir)
    y_test = np.load('%s/y_test.npy'%dataDir)
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=11)
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def build_wavenet(x, nclass, optimizer, loss):
    
    xshape= x.shape

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
    model.add(layers.Conv1D(nclass, 10, padding='same'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    model.add(layers.Reshape((nclass, )))
    model.add(layers.Activation('softmax'))
    
    model.compile(optimizer, loss, metrics=['accuracy'])    
    
    return model


MODELS = {
    'wavenet': build_wavenet,
}


def main():

    dataDir = sys.argv[1]
    model_name = sys.argv[2]

    # load data
    x_train, x_valid, _, y_train, y_valid, _ = load_data(dataDir)
    
    # load and draw model
    model = MODELS[model_name](x_train, 3, 'adam', 'sparse_categorical_crossentropy')
    model.summary()

if __name__=='__main__':
    main()
