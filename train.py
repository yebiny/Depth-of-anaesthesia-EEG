import os, sys, pickle, argparse, ast
from datetime import datetime
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from models import *
from draw import *

def set_save_path(opt):
    date = datetime.today().strftime("%Y_%m%d_%H%M")
    save_path= './results/%s_%s'%(date, opt['model_name'])
    if not os.path.isdir('results'): os.mkdir('results')
    if not os.path.isdir(save_path): os.mkdir(save_path)
    return save_path

def load_data(data_path, rs=34):
    x_data = np.load("%s/x_data.npy"%data_path)
    y_data = np.load("%s/y_data.npy"%data_path)
    l_data = np.load("%s/l_data.npy"%data_path)
    
    x, xv, y, yv = train_test_split(x_data, y_data, test_size=0.2, shuffle=True, random_state=rs)
    _, _, l, lv = train_test_split(x_data, l_data, test_size=0.2, shuffle=True, random_state=rs)
    return x, xv, y, yv, l, lv

def train(opt):
    save_path = set_save_path(opt) 
    with open('%s/opt.pickle'%save_path,'wb') as fw:
        pickle.dump(opt, fw) 

    # load data
    x_train, x_valid, y_train, y_valid, l_train, l_valid = load_data(opt['data_path'])
    print('trainset: ', x_train.shape, y_train.shape, l_train.shape)
    print('validset: ', x_valid.shape, y_valid.shape, l_valid.shape)

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
                    class_weight=np.array(opt['class_weights']),
                    callbacks = [reduce_lr, checkpoint, early_stop])

    # draw learning process
    draw_lprocess(history, save='%s/plot_lprocess'%save_path)

def set_option(model_name, data_path):
    option={
    'model_name': model_name,
    'data_path' : data_path,
    'epochs': 300,
    'class_weights' : [0.15,0.15,1,0.9],
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
