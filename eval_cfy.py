import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from train_cfy import load_data
class EVAL():
    
    def __init__(self, model_path, data_path):

        self.model =  load_model("%s/model.h5"%model_path)
        self.data_path = data_path

    def load_data(self):
        x_train = np.load('%s/x_train.npy'%self.data_path)
        l_train = np.load('%s/l_train.npy'%self.data_path)
        x_valid = np.load('%s/x_valid.npy'%self.data_path)
        l_valid = np.load('%s/l_valid.npy'%self.data_path)
        
        return x_train, l_train, x_valid, l_valid

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
        
        x_train, l_train, x_valid, l_valid = self.load_data()

        plt.figure(figsize=(12,4))
        
        ax = plt.subplot(1,2,1)
        l_pred = self.model.predict_classes(x_train)
        self._draw_cm(l_train, l_pred, ax, title='Train set' )
        
        ax = plt.subplot(1,2,2)
        l_pred = self.model.predict_classes(x_valid)
        self._draw_cm(l_valid, l_pred, ax, title='Valid set')
        
        if save!=None:
            plt.savefig(save)
        else: plt.show()
        plt.close('all')

def main():
    model_path = sys.argv[1]
    with open('%s/opt.pickle'%model_path, 'rb') as fr:
        opt = pickle.load(fr)
    data_path = opt['data_path']
    sys.stdout = open('%s/log.txt'%model_path,'w')
    print(opt)
    
    ev = EVAL(model_path, data_path)
    ev.draw_multi_cm('%s/plot_cm'%model_path)    
    

if __name__=='__main__':
    main()
