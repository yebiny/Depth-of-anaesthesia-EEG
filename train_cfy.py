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

def train(opt, x_data, y_data):
    save_path = set_save_path(opt) 
    with open('%s/opt.pickle'%save_path,'wb') as fw:
        pickle.dump(opt, fw) 

    # load data
    x_train, x_valid, y_train, y_valid = train_test_split( x_data, y_data
                                                         , test_size=0.15
                                                         , shuffle=True
                                                         , random_state=34)

    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

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
    history = model.fit(x_train, y_train,
                    batch_size=opt['batch_size'], epochs=opt['epochs'],
                    validation_data=(x_valid, y_valid),
                    class_weight=np.array(opt['class_weights']),
                    callbacks = [reduce_lr, checkpoint, early_stop])

    # draw learning process
    draw_lprocess(history, save='%s/plot_lprocess'%save_path)

def set_option( model_name, x_path, y_path, mask, ch
              , epochs=100
              , weights=[1,1,1,1]
              , batch_size=16
              , optimizer='adam'
              , nclass=4):
    
    option={
    'model_name': model_name,
    'x_path': x_path,
    'y_path': y_path,
    'mask': mask,
    'ch': ch,
    'epochs': epochs,
    'class_weights' : weights,
    'batch_size' : batch_size,
    'optimizer' : optimizer,
    'nclass': nclass
    }

    return option

def main():
    
    model_name= sys.argv[1]

    x_path = 'datasets_0708/try3/x_imfs.npy'
    y_path = 'datasets_0708/try3/y_data.npy'
    mask_path = 'datasets_0708/try3/mask_train_aug.npy'
    ch = 'concat'
    x_data, y_data = generate_datasets(x_path, y_path, mask_path, ch=ch)
    
    opt = set_option( model_name, x_path, y_path, mask_path, ch
                    , batch_size=16, weights=[0.7,0.8,1,1])
    print(opt)
    train(opt, x_data, y_data)
    
if __name__=='__main__':
    main()
