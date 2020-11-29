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

    def load_data(self):
        x_data = np.load('%s/x_data.npy'%self.dataDir)
        y_data = np.load('%s/y_data.npy'%self.dataDir)
        x_test = np.load('%s/x_test.npy'%self.dataDir)
        y_test = np.load('%s/y_test.npy'%self.dataDir)
        x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=11)
        return x_train, x_valid, x_test, y_train, y_valid, y_test

    def draw_cm(self, x_data, y_data, save=None):

        y_pred = self.model.predict_classes(x_data)
        cm = confusion_matrix(y_data, y_pred)
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax)
        buttom, top = ax.get_ylim()
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_ylim(buttom+0.5, top-0.5)
        
        if save!=None:
            plt.savefig(save)
        else: plt.show()
        plt.close('all')

def main():
    modelDir = sys.argv[1]
    
    ev = EVAL(modelDir)
    _, x_valid, x_test, _, y_valid, y_test = ev.load_data()
    ev.draw_cm(x_valid, y_valid, '%s/plot_cm_valid'%modelDir)
    ev.draw_cm(x_test, y_test, '%s/plot_cm_test'%modelDir)
 
if __name__=='__main__':
    main()
