import os, sys, pickle, argparse, ast
from datetime import datetime
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from models_reg import *
from draw import *

def set_save_path(opt):
    date = datetime.today().strftime("%m%d_%H%M")
    save_path= './results/21%s_%s_reg'%(date, opt['model_name'])
    if not os.path.isdir('results'): os.mkdir('results')
    if not os.path.isdir(save_path): os.mkdir(save_path)
    return save_path

def load_data(data_path, rs=34):
    x_train = np.load("%s/x_train.npy"%data_path)
    y_train = np.load("%s/y_train.npy"%data_path)
    x_valid = np.load("%s/x_valid.npy"%data_path)
    y_valid = np.load("%s/y_valid.npy"%data_path)
    
    return x_train, y_train, x_valid, y_valid

def train(opt):
    save_path = set_save_path(opt) 
    with open('%s/opt.pickle'%save_path,'wb') as fw:
        pickle.dump(opt, fw) 

    # load data
    x_train, l_train, x_valid, l_valid = load_data(opt['data_path'])
    print('trainset: ', x_train.shape, l_train.shape)
    print('validset: ', x_valid.shape, l_valid.shape)

    # load model
    if opt['model_name'] in MODELS:
          model = MODELS[opt['model_name']](x_train.shape[1:], opt['optimizer'])
    else: model = load_model(model_name)
    plot_model(model, show_shapes=True, to_file='%s/model.png'%save_path)
    
    # callbacks
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    checkpoint = ModelCheckpoint("%s/model.h5"%save_path, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=8)
    
    # train 
    history = model.fit(x_train, l_train ,
                    batch_size=opt['batch_size'], epochs=opt['epochs'],
                    validation_data=(x_valid, l_valid),
                    callbacks = [reduce_lr, checkpoint, early_stop])

    # draw learning process
    draw_lprocess(history, save='%s/plot_lprocess'%save_path)

def set_option(model_name, data_path):
    option={
    'model_name': model_name,
    'data_path' : data_path,
    'epochs': 100,
    'batch_size' : 16,
    'optimizer' : 'adam'
    }

    return option

def main():
    
    model_name= sys.argv[1]

    if model_name in MODELS:
        data_path = sys.argv[2]
        opt = set_option(model_name, data_path)

    else: 
        with open('%s/opt.pickle'%model_name, 'rb') as fr:
            opt = pickle.load(fr)

    print(opt)
    train(opt)
    
if __name__=='__main__':
    main()
