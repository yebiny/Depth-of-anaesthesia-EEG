import os, sys, glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

def _get_eeg_bis_from_mat(matfile):
    f = h5py.File(matfile, mode='r')
    data={}
    for a, b in f.items():
        data[a]=np.array(b)
    eeg = data['EEG'].ravel()
    bis = data['bis'].ravel()
    return eeg, bis

def _get_time_arr(eeg, bis, eeg_ratio=1/125, bis_ratio=5):
    eegt = np.array([(eeg_ratio)*i for i in range(len(eeg))])
    bist = np.array([(bis_ratio)*i for i in range(len(bis))])
    return eegt, bist

def plot_bis_with_range(mat_file, pointer=[], title=None):

    eeg, bis = _get_eeg_bis_from_mat(mat_file)
    eegt, bist = _get_time_arr(eeg, bis)
    print(eeg.shape, eegt.shape, bis.shape, bist.shape)

    fig = plt.figure(figsize=(12,5))
    if title: plt.title(title)
    plt.plot(bist, bis,  marker='.')
    plt.xlabel("Time")

    for i in [40, 60, 65, 80, 85]:
        plt.plot((0,bist[-1]),(i,i), color='red', alpha=0.5)
        plt.text(-100, i, str(i))

    for p in pointer:
        plt.plot((bist[p],bist[p]),(0,100), color='green', alpha=0.5)
        plt.text(bist[p], bis[p]+3, "%i"%(bis[p]))
        plt.text(bist[p], 0, "%i"%(bist[p]))

    plt.show()
