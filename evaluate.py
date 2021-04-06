import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
class EVAL():
    
    def __init__(self, model_path, data_path):

        self.model =  load_model("%s/model.h5"%model_path)
        self.data_path = data_path

    def load_data(self):
        x_valid = np.load('%s/x_valid.npy'%self.data_path)
        l_valid = np.load('%s/l_valid.npy'%self.data_path)
        x_test = np.load('%s/x_test.npy'%self.data_path)
        l_test = np.load('%s/l_test.npy'%self.data_path)
        
        return x_valid, l_valid, x_test, l_test

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
        
        x_valid, l_valid, x_test, l_test = self.load_data()

        plt.figure(figsize=(12,4))
        
        l_pred = self.model.predict_classes(x_valid)
        ax = plt.subplot(1,2,1)
        self._draw_cm(l_valid, l_pred, ax, title='Valid set')
        
        l_pred = self.model.predict_classes(x_test)
        ax = plt.subplot(1,2,2)
        self._draw_cm(l_test, l_pred, ax, title='Test set' )
        
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
    if ev.model.output.shape[1] == 1:
        ev.draw_response('%s/plot_response'%model_path)
        ev.draw_roc('%s/plot_roc'%model_path)
    

if __name__=='__main__':
    main()
