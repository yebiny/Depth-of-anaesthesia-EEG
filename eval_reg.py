import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn import metrics
import seaborn as sns
class EVAL():
    
    def __init__(self, model_path, data_path):

        self.model =  load_model("%s/model.h5"%model_path)
        self.data_path = data_path
        self.x_train, self.y_train, self.l_train = self.load_data('train')
        self.x_valid, self.y_valid, self.l_valid = self.load_data('valid')

        self.y_pred = self.model.predict(self.x_train)
        self.y_v_pred = self.model.predict(self.x_valid)
   
    def draw_reg(self, save):
        plt.figure(figsize=(6,6))
        plt.scatter(self.y_valid, self.y_v_pred, alpha=0.6, marker='.')
        plt.xlabel("True")
        plt.ylabel("Pred")
        if save!=None:
            plt.savefig(save)
        else: plt.show()
        plt.close('all')

    def load_data(self, data_type):
        if data_type=='valid':
            x = np.load('%s/x_valid.npy'%self.data_path)
            y = np.load('%s/y_valid.npy'%self.data_path)
            l = np.load('%s/l_valid.npy'%self.data_path)
        if data_type=='train':
            x = np.load('%s/x_train.npy'%self.data_path)
            y = np.load('%s/y_train.npy'%self.data_path)
            l = np.load('%s/l_train.npy'%self.data_path)
        x=x/50
        return x, y, l

    def _draw_cm(self, y_true, y_pred, ax, title='Confusion matrix'):
        cm = metrics.confusion_matrix(y_true, y_pred)
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm, annot=True, ax = ax)
        plt.title(title)
        buttom, top = ax.get_ylim()
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_ylim(buttom+0.5, top-0.5)

    def draw_multi_cm(self, cut, save=None):
        
        l_pred = []
        for y in self.y_pred:
            if y<cut: l_pred.append(0)
            else: l_pred.append(1)

        plt.figure(figsize=(12,4))
        ax = plt.subplot(1,2,1)
        self._draw_cm(self.l_train, l_pred, ax, title='Train set')
        
        l_pred = []
        for y in self.y_v_pred:
            if y<cut: l_pred.append(0)
            else: l_pred.append(1)
        
        ax = plt.subplot(1,2,2)
        self._draw_cm(self.l_valid, l_pred, ax, title='Valid set')
        
        if save!=None:
            plt.savefig(save)
        else: plt.show()
        plt.close('all')

def main():
    model_path = sys.argv[1]
    cm_cut = float(sys.argv[2])
    with open('%s/opt.pickle'%model_path, 'rb') as fr:
        opt = pickle.load(fr)
    data_path = opt['data_path']
    sys.stdout = open('%s/log.txt'%model_path,'w')
    print(opt)
    
    ev = EVAL(model_path, data_path)
    ev.draw_reg('%s/plot_reg'%model_path) 
    ev.draw_multi_cm(cm_cut, '%s/plot_cm'%model_path)    
    

if __name__=='__main__':
    main()
