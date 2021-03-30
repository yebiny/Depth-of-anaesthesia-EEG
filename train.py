import os, sys, pickle, argparse
from datetime import datetime
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from models import *
from draw import *

def save_option(args, save_path):
    log='''
    data_dir : {d}
    model_name : {m}
    epochs : {e}
    batch_size : {b}
    optimizer  : {o}
    '''.format( d=args.data_path
              , m=args.model_name
              , e=args.epochs
              , b=args.batch_size
              , o=args.optimizer)
    
    with open('%s/option.txt'%save_path, 'w') as f:
        f.write(log)
    

def set_save_directory():
    date = datetime.today().strftime("%Y_%m%d_%H%M")
    save_path= './results/%s'%date
    if not os.path.isdir('results'): os.mkdir('results')
    if not os.path.isdir(save_path): os.mkdir(save_path)
    return save_path

def get_model(model_name, x_train):
    return model

def set_callbacks(save_path):
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    checkpoint = ModelCheckpoint("%s/model.h5"%save_path, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=8)
    
    return [reduce_lr, checkpoint, early_stop]

def load_data(data_path, rs=34):
    x_data = np.load("%s/x_data.npy"%data_path)
    y_data = np.load("%s/y_data.npy"%data_path)
    l_data = np.load("%s/l_data.npy"%data_path)
    
    x, xv, y, yv = train_test_split(x_data, y_data, test_size=0.2, shuffle=True, random_state=rs)
    _, _, l, lv = train_test_split(x_data, l_data, test_size=0.2, shuffle=True, random_state=rs)
    return x, xv, y, yv, l, lv

def train(args):
    save_path = set_save_directory() 
    save_option(args, save_path)
    
    # load data
    x_train, x_valid, y_train, y_valid, l_train, l_valid = load_data(args.data_path)
    print('trainset: ', x_train.shape, y_train.shape, l_train.shape)
    print('validset: ', x_valid.shape, y_valid.shape, l_valid.shape)

    # load model
    if args.model_name in MODELS:
          model = MODELS[args.model_name](x_train.shape[1:], args.optimizer)
    else: model = load_model(model_name)
    plot_model(model, show_shapes=True, to_file='%s/model.png'%save_path)

    # train
    callbacks = set_callbacks(save_path)
    history = model.fit(x_train, y_train ,
                        batch_size=args.batch_size, epochs=args.epochs,
                        validation_data=(x_valid, y_valid),
                        #class_weight=np.array([args.class_weights, 1]),
                        callbacks = callbacks
                       )

    # draw learning process
    draw_lprocess(history, save='%s/plot_lprocess'%save_path)


def parse_args():
    opt = argparse.ArgumentParser(description="==== Training ====")
    opt.add_argument(dest='data_path', type=str, help=': data directory ')
    opt.add_argument(dest='model_name', type=str, help=': select model ')
    opt.add_argument('-e', dest='epochs', type=int, default=50, help=': epochs(default: 50)')
    opt.add_argument('-b', dest='batch_size', type=int, default=16, help=': batch_size(default: 16)') 
    opt.add_argument('-o', dest='optimizer', type=str, default='adam', help=': optimzier(default: adam)')
    args = opt.parse_args()

    return args


def main():
    args = parse_args()
    train(args)
    
if __name__=='__main__':
    main()
