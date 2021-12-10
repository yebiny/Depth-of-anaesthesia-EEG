import glob, os, sys
import numpy as np
import h5py
from PyEMD import EMD
from scipy import signal

class DoAGenerator():
    def __init__(self, window_sec=10, step_sec=5, bis_sr=0.2, eeg_sr=125):
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.bis_sr = bis_sr
        self.eeg_sr = eeg_sr

    def get_eeg_bis_from_mat(self, f_name):
        f = h5py.File(f_name, mode='r')
        data={}
        for a, b in f.items():
            data[a]=np.array(b)
        eeg = data['EEG'].ravel()
        bis = data['bis'].ravel()
        
        return f_name, eeg, bis
    
    def split_eeg_by_bis(self, eeg, bis):
        eeg_arr, bis_arr = [],[]
        for i, b in enumerate(bis):
            e_start = i*(self.step_sec*self.eeg_sr)
            e = eeg[e_start:e_start+(self.window_sec*self.eeg_sr)]
            if len(e) != self.window_sec*self.eeg_sr: continue
            eeg_arr.append(e)
            bis_arr.append(b)
        return np.array(eeg_arr), np.array(bis_arr)
    
    def filter_eegs(self, x):
        filter = signal.firwin(400, [0.01, 0.6], pass_zero=False)
        y = signal.convolve(x, filter, mode='same')
        return np.array(y)
    
    def get_imf(self, x):
        t = np.linspace(0, self.window_sec, self.window_sec*self.eeg_sr)
        imfs = EMD().emd(x, t)
        imfs = imfs[:3]
        return np.array(imfs)
    
    def generate_ds_from_paths(self, f_list):
    
        x_data, y_data = [], []
        for f in f_list:
            f_name, eeg, bis = self.get_eeg_bis_from_mat(f)
            print("%s, eeg: %i, bis : %i, ratio: %f"%
                  (f_name, len(eeg), len(bis), len(eeg)/len(bis)))
    
            eeg_arr, bis_arr = self.split_eeg_by_bis(eeg, bis)
            for eeg in eeg_arr:   
                eeg_filtered = self.filter_eegs(eeg)
                eeg_imfs = self.get_imf(eeg_filtered)
                x_data.append(eeg_imfs)
            y_data.extend(bis_arr)
        
        print("END ROOP")
        y_data = np.array(y_data)
        x_data = np.array(x_data) 
        return x_data, y_data

def main():

    DATA_DIR = sys.argv[1]
    SAVE_DIR = sys.argv[2]

    f_list = sorted(glob.glob(DATA_DIR+'/*.mat'))
    print(len(f_list), f_list[0])
    
    doaGene = DoAGenerator() 
    x_data, y_data = doaGene.generate_ds_from_paths(f_list[:1])
    print("End process: ", x_data.shape, y_data.shape)

    np.save('%s/x_data.npy'%SAVE_DIR, x_data)
    np.save('%s/y_data.npy'%SAVE_DIR, y_data)

if __name__=='__main__':
    main()
