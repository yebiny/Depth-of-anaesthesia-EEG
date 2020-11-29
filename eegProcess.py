import os, sys, glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from PyEMD import EMD


class EEGProcess():

    def __init__(self, dataDir, sec, window_size):
        self.file_list = glob.glob('%s/*mat'%dataDir)
        self.sec = sec
        self.window_size=window_size
        self.n_files = len(self.file_list)
        print('Total %i files : '%self.n_files, self.file_list) 
        
    def process(self, n):
        xset, yset, idxs = [], [], []
        for f in self.file_list[:n]:
            eeg, bis, idx = self.get_data(f)
            print('===',f,'=== : ',  len(bis))
            r, need_len = self.get_data_lenth(eeg, bis)
            eeg = self.cut_eeg(eeg, bis, need_len)

            eeg_sp, bis_sp = self.split_eeg(eeg, bis, r)
            eeg_filterd = self.filter_eegs(eeg_sp)
            imfs = self.get_imf(eeg_filterd)
            print(imfs.shape, bis_sp.shape)

            xset.extend(imfs)
            yset.extend(bis_sp)
            idxs.extend(np.array([idx for i in range(len(bis_sp))]))

        xset = np.array(xset)
        yset = np.array(yset)
        idxs = np.array(idxs)
        print('Final output:', xset.shape, yset.shape, idxs.shape)
        return xset, yset, idxs

    def get_data(self, filepath):
        idx = int(filepath.split('/')[-1].split('.')[0].split('case')[-1])
        f = h5py.File(filepath)
        data={}
        for a, b in f.items():
            data[a]=np.array(b)
        eeg = data['EEG']
        bis = data['bis']
        eeg = eeg.ravel()
        bis = bis.ravel()
        
        return eeg, bis, idx
    
    def get_data_lenth(self, eeg, bis):
        r = int(np.trunc(len(eeg)/len(bis)))
        for i in range(len(bis)-1):
            start = i*r
            end = start+self.window_size
        #print('* ratio: ', r, 'need len: ', end, 'eeg len: ', len(eeg))
        
        return r, end
    
    def cut_eeg(self, eeg, bis, need_len):
        n_rm =len(eeg)-need_len
        a = int(np.trunc(n_rm/2))
        b = int(np.ceil(n_rm/2))
        c = len(eeg)-b 
        eeg_new = eeg[a:c]
        return eeg_new
    
    def split_eeg(self, eeg, bis, r):
        eeg_splited = []
        for i in range(len(bis)-1):
            start = int(i*r)
            end = int(start+self.window_size)
            x = eeg[start:end]
            eeg_splited.append(x)
            #print(i, len(x), len(eeg_splited))
        eeg_splited = np.array(eeg_splited)
        bis_splited = bis[:-1]
    
        return eeg_splited, bis_splited
    
    def filter_eegs(self, eegs):
        outs=[]
        for i, eeg in enumerate(eegs):
            filter = signal.firwin(400, [0.01, 0.6], pass_zero=False)
            y = signal.convolve(eeg, filter, mode='same')
            outs.append(y)
        outs = np.array(outs)
        return outs
    
    
    def get_imf(self, eegs):
        t = np.linspace(0, self.window_size/self.sec, self.window_size)
        outs=[]
        for i, eeg in enumerate(eegs):
            imfs = EMD().emd(eeg, t)
            imfs = imfs[:3]
            outs.append(imfs)
            if i%200==0: print(i)
        outs = np.array(outs)
        return outs
    

def main():
    dataDir = sys.argv[1]
    saveDir = sys.argv[2]
    if not os.path.isdir(saveDir): os.mkdir(saveDir)
    else: 
        print('! Already exist')
        exit()

    SECOND = 125
    WINDOW_SIZE = 10*SECOND

    EP = EEGProcess(dataDir, SECOND, WINDOW_SIZE)
    xset, yset, idx = EP.process(EP.n_files)

    np.save('%s/xset'%saveDir, xset)
    np.save('%s/yset'%saveDir, yset)
    np.save('%s/idx'%saveDir, idx)


if __name__ == '__main__':
    main()
