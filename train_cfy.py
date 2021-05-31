import os, sys, pickle, argparse, ast
from datetime import datetime
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from models_cfy import *
from draw import *
from load_data import *

def set_save_path(opt):
    date = datetime.today().strftime("%m%d%H%M")
    save_path= './results/%s'%date
    if not os.path.isdir('results'): os.mkdir('results')
    if not os.path.isdir(save_path): os.mkdir(save_path)
    return save_path

def _load_data(data_path):
    x_train = np.load("%s/x_train.npy"%data_path)
    y_train = np.load("%s/y_train.npy"%data_path)
    x_valid = np.load("%s/x_valid.npy"%data_path)
    y_valid = np.load("%s/y_valid.npy"%data_path)
    #x_train, x_valid = np.expand_dims(x_train, 2), np.expand_dims(x_valid, 2)
    print(x_train.shape, x_valid.shape)
    x_train, x_valid = x_train/50, x_valid/50
    return x_train, y_train, x_valid, y_valid

def train(opt):
    save_path = set_save_path(opt) 
    with open('%s/opt.pickle'%save_path,'wb') as fw:
        pickle.dump(opt, fw) 

    # load data
    ld = LoadDataset(opt['data_path'], opt['xlen'])
    x_list, y_list = ld.process_for_cfy()
    x_train, x_valid = x_list[:2]
    y_train, y_valid = y_list[:2]

    print('trainset: ', x_train.shape, y_train.shape)
    print('validset: ', x_valid.shape, y_valid.shape)

    # load model
    if opt['model_name'] in MODELS:
          model = MODELS[opt['model_name']](x_train.shape[1:], opt['nclass'], opt['optimizer'])
    else: model = load_model(opt['model_name']+'/model.h5')
    plot_model(model, show_shapes=True, to_file='%s/model.png'%save_path)
    
    # callbacks
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    checkpoint = ModelCheckpoint("%s/model.h5"%save_path, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    # train 
    history = model.fit(x_train, y_train ,
                    batch_size=opt['batch_size'], epochs=opt['epochs'],
                    validation_data=(x_valid, y_valid),
                    #class_weight=np.array(opt['class_weights']),
                    callbacks = [reduce_lr, checkpoint, early_stop])

    # draw learning process
    draw_lprocess(history, save='%s/plot_lprocess'%save_path)

def set_option(model_name, data_path, x_sec):
    option={
    'model_name': model_name,
    'data_path' : data_path,
    'xlen': 125*int(x_sec),
    'epochs': 100,
    #'class_weights' : [0.4,0.4, 1, 0.9],
    'batch_size' : 16,
    'optimizer' : 'adam',
    'nclass':4
    }

    return option

def main():
    
    model_name= sys.argv[1]
    data_path = sys.argv[2]
    x_sec = sys.argv[3]

    opt = set_option(model_name, data_path, x_sec)

    print(opt)
    train(opt)
    
if __name__=='__main__':
    main()
