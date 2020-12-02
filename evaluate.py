import os, sys, pickle
import numpy as np

from drawTools import *
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

class EVAL():
    
    def __init__(self, modelDir):

        with open('%s/opt.pickle'%modelDir, 'rb') as fr:
            opt = pickle.load(fr)
        print(opt)
        self.dataDir = opt['dataDir']
        self.modelDir = modelDir

        self.model =  load_model("%s/model.h5"%modelDir)
        _, self.x_valid, self.x_test, _, self.y_valid, self.y_test = self.load_data()

    def load_data(self):
        x_data = np.load('%s/x_data.npy'%self.dataDir)
        y_data = np.load('%s/y_data.npy'%self.dataDir)
        x_test = np.load('%s/x_test.npy'%self.dataDir)
        y_test = np.load('%s/y_test.npy'%self.dataDir)
        x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=11)
        return x_train, x_valid, x_test, y_train, y_valid, y_test

    def draw_cm(self, save=None):

        plt.figure(figsize=(12,4))
        
        y_pred = self.model.predict_classes(self.x_valid)
        ax = plt.subplot(1,2,1)
        draw_cm(self.y_valid, y_pred, ax, title='validset')
        
        y_pred = self.model.predict_classes(self.x_test)
        ax = plt.subplot(1,2,2)
        draw_cm(self.y_test, y_pred, ax, title='testset' )
        
        if save!=None:
            plt.savefig(save)
        else: plt.show()
        plt.close('all')

    def get_response(self, x_data, y_data):
        
        y_pred = self.model.predict(x_data)
        is_sig = y_data.astype(np.bool)
        is_bkg = np.logical_not(y_data)
        pred_true = y_pred[is_sig]
        pred_false= y_pred[is_bkg]
        
        return pred_true, pred_false

    def draw_response(self, save=None):
        
        v_true, v_false = self.get_response(self.x_valid, self.y_valid)
        t_true, t_false = self.get_response(self.x_test, self.y_test) 

        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        draw_response(v_true, v_false, title='validset')
        plt.subplot(1,2,2)
        draw_response(t_true, t_false, title='testset')
        if save!=None:
            plt.savefig(save)
        else: plt.show()

    def draw_roc(self, save=None):
        
        plt.figure(figsize=(12,5))
        
        ax = plt.subplot(1,2,1)
        y_pred = self.model.predict(self.x_valid)
        fpr, tpr, _ = metrics.roc_curve(self.y_valid , y_pred[:, 0])
        draw_roc(fpr, tpr, ax, title='ROC curve, validset')
        
        ax = plt.subplot(1,2,2)
        y_pred = self.model.predict(self.x_test)
        fpr, tpr, _ = metrics.roc_curve(self.y_test , y_pred[:, 0])
        draw_roc(fpr, tpr, ax, title='ROC curve, testset')
        
        if save!=None:
            plt.savefig(save)
        else: plt.show()

def main():
    modelDir = sys.argv[1]
    sys.stdout = open('%s/log.txt'%modelDir,'w')
    
    ev = EVAL(modelDir)
    ev.draw_cm('%s/plot_cm'%modelDir)    
    if ev.model.output.shape[1] == 1:
        ev.draw_response('%s/plot_response'%modelDir)
        ev.draw_roc('%s/plot_roc'%modelDir)

if __name__=='__main__':
    main()
