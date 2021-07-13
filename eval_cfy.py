import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from load_data import *

class EVAL():
    
    def __init__(self, model_path, x_test, y_test):

        self.model =  load_model("%s/model.h5"%model_path)
        self.x_test = x_test
        self.y_test = y_test

    def _draw_cm(self, y_true, y_pred, ax, title='Confusion matrix'):
        cm = metrics.confusion_matrix(y_true, y_pred)
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm, annot=True, ax = ax)
        plt.title(title)
        buttom, top = ax.get_ylim()
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_ylim(buttom+0.5, top-0.5)

    def draw_multi_cm(self, save=None):
        # load data
        #ld = LoadDataset(self.data_path, self.xlen)
        #x_list, y_list = ld.process_for_cfy()
        #x_train, x_valid, x_test = x_list
        #y_train, y_valid, y_test = y_list
        
        plt.figure(figsize=(18,4))
        
        ax = plt.subplot(1,3,1)
        y_pred = self.model.predict_classes(self.x_test)
        self._draw_cm(self.y_test, y_pred, ax, title='TRAIN set')
        
        ax = plt.subplot(1,3,2)
        y_pred = self.model.predict_classes(self.x_test)
        self._draw_cm(self.y_test, y_pred, ax, title='VALID set' )
        
        ax = plt.subplot(1,3,3)
        y_pred = self.model.predict_classes(self.x_test)
        self._draw_cm(self.y_test, y_pred, ax, title='TEST set')
        
        
        if save!=None:
            plt.savefig(save)
        else: plt.show()
        plt.close('all')

def main():
    model_path = sys.argv[1]

    x_path = 'datasets_0708/try3/x_imfs.npy'
    y_path = 'datasets_0708/try3/y_data.npy'
    mask_path = 'datasets_0708/try3/mask_test.npy'
    ch = 'concat'
    x_test, y_test = generate_datasets(x_path, y_path, mask_path, ch=ch)
    
    with open('%s/opt.pickle'%model_path, 'rb') as fr:
        opt = pickle.load(fr)
    sys.stdout = open('%s/log.txt'%model_path,'w')
    print(opt)
    
    ev = EVAL(model_path, x_test, y_test)
    ev.draw_multi_cm('%s/plot_cm'%model_path)    
    

if __name__=='__main__':
    main()
