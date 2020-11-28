import os, sys, pickle
import numpy as np

from datetime import datetime
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from models import *
from drawTools import *

def load_data(dataDir):
    x_data = np.load('%s/x_data.npy'%dataDir)
    y_data = np.load('%s/y_data.npy'%dataDir)
    x_test = np.load('%s/x_test.npy'%dataDir)
    y_test = np.load('%s/y_test.npy'%dataDir)
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=11)
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def train(opt):
    # make save directory
    date = datetime.today().strftime("%Y_%m%d_%H%M")
    saveDir= './1-results/%s'%date
    if not os.path.isdir(saveDir): os.mkdir(saveDir)
    
    # get options
    dataDir = opt['dataDir']
    model = opt['model']
    epochs =  opt['epochs']
    class_weights = opt['class_weights']
    batch_size = opt['batch_size']
    
    # save options
    with open('%s/opt.pickle'%saveDir,'wb') as fw:
        pickle.dump(opt, fw)
    
    # load data
    x_train, x_valid, _, y_train, y_valid, _ = load_data(dataDir)
    
    # load and draw model
    model = MODELS[model](x_data, 3, 'adam', 'sparse_categorical_crossentropy')
    plot_model(model, show_shapes=True, to_file='%s/model.png'%saveDir)
    
    # callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    model_checkpoint_cb = ModelCheckpoint("%s/model.h5"%saveDir, save_best_only=True)
    
    # train
    history = model.fit(x_train, y_train ,
                        batch_size=batch_size, epochs=epochs,
                        validation_data=(x_valid, y_valid),
                        class_weight=class_weights,
                        callbacks = [model_checkpoint_cb, reduce_lr]
    
    )

    # draw learning process
    draw_lprocess(history, save='%s/plot_lprocess'%saveDir)


def main():
    opt={
    'dataDir' : sys.argv[1],
    'model' : 'wavenet',
    'epochs': 20,
    #'class_weights' : np.array([0.68741815, 0.68741815, 1]),
    'class_weights' : np.array([0.78, 0.78, 1]),
    'batch_size' : 64,
    }
    
    print(opt)
    train(opt)
    
if __name__=='__main__':
    main()
