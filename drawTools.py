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

def draw_multi_hist(data_list, n_row=1, save=None):
    n_data = len(data_list)
    plt.figure(figsize=((n_data/n_row)*3, n_row*3))
    for i, data in enumerate(data_list):
        plt.subplot(n_row, n_data/n_row, i+1)
        plt.title(data)
        plt.hist(data_list[data], color='#9467bd', rwidth=0.9, bins=3, alpha=0.6)
    
    if save!=None:
        plt.savefig(save)
    else:
        plt.show()

def draw_cm(y_true, y_pred, ax, title='Confusion matrix'):    
     cm = confusion_matrix(y_true, y_pred)
     cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
     sns.heatmap(cm, annot=True, ax = ax)
     plt.title(title)
     buttom, top = ax.get_ylim()
     ax.set_xlabel("Pred")
     ax.set_ylabel("True")
     ax.set_ylim(buttom+0.5, top-0.5)

def draw_response(sig, bkg, title='Response'):
    plt.title(title)
    kwargs = dict(histtype='stepfilled', alpha=0.4, density=True, bins=25)
    plt.hist(sig, **kwargs, edgecolor='b')
    plt.hist(bkg, **kwargs, edgecolor='r')
    plt.legend(['awake', 'anaesthesia'], loc='upper center')
    
