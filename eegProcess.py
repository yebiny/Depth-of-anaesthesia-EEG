import os, glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from math import pi
import pandas as pd
from PyEMD import EMD

def get_data(filepath):
    f = h5py.File(filepath)
    data={}
    for a, b in f.items():
        data[a]=np.array(b)
    eeg = data['EEG']
    bis = data['bis']
    return eeg, bis

def split_eeg(eeg, bis, t1=125, window=3750, step=625):
    n = len(eeg)//step
    y=[]
    for i in range(n):
        start = i*step
        end = start+window
        x = eeg[start:end]
        if len(x)!=window:
            continue
            
        y.append(x)
    y = np.array(y)
    bis = bis[:len(y)]
    
    print(y.shape, bis.shape)
    return y, bis

def sampen(L, m, r):
    N = len(L)
    B = 0.0
    A = 0.0

    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    
    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    # Return SampEn
    return -np.log(A / B)

def get_sampen(eegs):
    outs=[]
    for i, eeg in enumerate(eegs):
        y = sampen(eeg, 2, 0.1*np.std(eeg))
        outs.append(y)
        if i%10==0: print(i)

    return outs

def filtering(eegs):
    outs=[]
    for i, eeg in enumerate(eegs):
        filter = signal.firwin(400, [0.01, 0.6], pass_zero=False)
        y = signal.convolve(eeg, filter, mode='same')
        outs.append(y)
        if i%200==0: print(i)
    outs = np.array(outs)
    print(outs.shape)
    return outs

def get_imf(eegs):
    window, t1 = 3750, 125
    t = np.linspace(0, window/t1, window)
    outs=[]
    for i, eeg in enumerate(eegs):
        imf = EMD().emd(eeg, t)
        outs.append([imf[0], imf[1]])
        if i%100==0: print(i)
    outs = np.array(outs)
    print(outs.shape)
    return outs

def main():

    file_list = glob.glob('0-data/org/case16.mat')
    print('* data list : ',file_list)
    xset, yset = [], []
    for i, f in enumerate(file_list):
        print('Start %ith file : %s'%(i,f))
        eeg, bis = get_data(f)
        eeg, bis = eeg[0], bis[0]
    
        print('* split')
        split_x, y= split_eeg(eeg, bis)
        print('* filter')
        filter_x = filtering(split_x)
        print('* imf')
        imfs_x = get_imf(filter_x)
    
        xset.extend(imfs_x)
        yset.extend(y)
    
    xset = np.array(xset)
    yset = np.array(yset)
    np.save('xset', xset)
    np.save('yset', yset)
    
    print(xset.shape, yset.shape)
    
if __name__ == '__main__':
    main()      
