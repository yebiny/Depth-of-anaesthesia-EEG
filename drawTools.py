import os, sys, pickle
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def draw_lprocess(history, save=None):
    figure=plt.figure()
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(history.epoch)
    plt.plot(history.history['loss'], 'y', label='t loss')
    plt.plot(history.history['val_loss'], 'r', label='v loss')
    plt.legend(loc='upper right')
    if save!=None:
        plt.savefig(save+'_loss')        
    else:
        plt.show()
    
    figure=plt.figure()
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xticks(history.epoch)
    plt.plot(history.history['accuracy'], 'y', label='t acc')
    plt.plot(history.history['val_accuracy'], 'r', label='v acc')
    plt.legend(loc='lower right')
    if save!=None:
        plt.savefig(save+'_acc')        
    else:
        plt.show()



def draw_cm(x_data, y_data, model):

    y_pred = model.predict_classes(x_data)
    cm = confusion_matrix(y_data, y_pred)
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
