import os, sys, pickle
import numpy as np
import argparse
from datetime import datetime
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from models import *
from drawTools import *

def train(args):
    # make save directory
    date = datetime.today().strftime("%Y_%m%d_%H%M")
    save_path= './results/%s'%date
    if not os.path.isdir(save_path): os.mkdir(save_path)
    
    save_option(args, save_path)
    
    #with open('%s/opt.pickle'%save_path,'wb') as fw:
    #    pickle.dump(opt, fw)
    
    # load data
    x_train, x_valid, _, y_train, y_valid, _ = load_data(args.data_path)
    print(x_train.shape, y_train.shape)

    # load and draw model
    if args.model_name in MODELS:
        model = MODELS[args.model_name](x_train, args.activation)
    else:  model = load_model(args.model_name)
    plot_model(model, show_shapes=True, to_file='%s/model.png'%save_path)
    
    # callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    model_checkpoint_cb = ModelCheckpoint("%s/model.h5"%save_path, save_best_only=True)
    
    # train
    history = model.fit(x_train, y_train ,
                        batch_size=args.batch_size, epochs=args.epochs,
                        validation_data=(x_valid, y_valid),
                        class_weight=np.array([args.class_weights, 1]),
                        callbacks = [model_checkpoint_cb, reduce_lr]
    
    )

    # draw learning process
    draw_lprocess(history, save='%s/plot_lprocess'%save_path)

def save_option(args, save_path):
    opt={
    'dataDir' : args.data_path,
    'model_name' : args.model_name,
    'epochs': args.epochs,
    'class_weights' : np.array([args.class_weights, 1]),
    'batch_size' : args.batch_size,
    'activation' : args.activation
    }
    # save options
    f = open("%s/opt.txt"%save_path, "w")
    for o in opt:
        f.write(o, opt[o])
    f.close()
    print(opt)

def parse_args():
    opt = argparse.ArgumentParser(description="==== Training ====")
    opt.add_argument(dest='data_path', type=str, help=': data directory ')
    opt.add_argument(dest='model_name', type=str, help=': select model ')
    opt.add_argument('-e', dest='epochs', type=int, default=50, help=': epochs(default: 50)')
    opt.add_argument('-b', dest='batch_size', type=int, default=64, help=': batch_size(default: 64)') 
    opt.add_argument('-w', dest='class_weights', type=float, default=0.74, help=': weight for zero(default: 0.74)')
    opt.add_argument('-a', dest='activation', type=str, default='adam', help=': activation(default: adam)')
    args = opt.parse_args()

    return args


def main():
    args = parse_args()
    train(args)
    
if __name__=='__main__':
    main()
