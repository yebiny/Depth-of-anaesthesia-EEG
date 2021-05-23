import numpy as np
from PyEMD import EMD
from scipy import signal
import h5py
import glob
from sklearn.model_selection import train_test_split

class DataProcess():

    def __init__( self
                , step_sec
                , window_sec
                , eeg_per_sec=125
                , bis_per_sec=0.2):
        
        # setting
        self.step_sec = step_sec 
        self.window_sec = window_sec
        self.eeg_per_sec = eeg_per_sec
        self.bis_per_sec = bis_per_sec
        
        self.eeg_step = self.step_sec*self.eeg_per_sec
        self.eeg_window = self.window_sec*self.eeg_per_sec
    
    def get_eeg_bis_from_mat(self, mat_file):
        f = h5py.File(mat_file, mode='r')
        data={}
        for a, b in f.items():
            data[a]=np.array(b)
        eeg = data['EEG'].ravel()
        bis = data['bis'].ravel()
        return eeg, bis
        
    def get_eeg_bis_time(self, eeg, bis):
        eeg_tr, bis_tr = 1/self.eeg_per_sec, 1/self.bis_per_sec

        bis_start = np.ceil(self.window_sec/bis_tr)*bis_tr
        eeg_t = np.array([ eeg_tr*i for i in range(len(eeg)) ])
        bis_t = np.array([ bis_tr*i+bis_start for i in range(len(bis)) ])
        return eeg_t, bis_t
    
    def cut_eeg_evenly(self, x, t):
        assert len(x)==len(t)
        
        x_out, t_out = [], []
        n_step = int(np.trunc(len(x)/self.eeg_step))
        for step in range(n_step):
            start = step*self.eeg_step
            x_cut = x[start:start+self.eeg_window]
            t_cut = t[start:start+self.eeg_window]
            if (len(x_cut)==(self.eeg_window)):
                x_out.append(x_cut)
                t_out.append(t_cut)
    
        return np.array(x_out), np.array(t_out)
    
    def process_eeg(self, x_arrs, t_data):
        assert x_arrs.shape == t_data.shape
    
        x_out = []
        for i, (x, t) in enumerate(zip(x_arrs, t_data)):
            filter = signal.firwin(400, [0.01, 0.6], pass_zero=False)
            x = signal.convolve(x, filter, mode='same')
            x = EMD().emd(x, t)
            x = x[0]+x[1]
            x_out.append(x)
    
        return np.array(x_out)
    
    def process_norm(self, x_arrs):
        size = x_arrs.shape[0]*x_arrs.shape[1]
        x = np.reshape(x_arrs, (size))
        mean = np.mean(x)
        std = np.std(x)
    
        x_out = (x_arrs-mean)/std
        return x_out
    
    def process_label(self, mat_file, bis, bis_t, x_data, t_data):
        tr = 1/self.bis_per_sec

        x_out, y_out = [], []
        subid = mat_file.split('case')[-1].split('.')[0]
        for x, t in zip(x_data, t_data):
            tend = np.ceil(t[-1]/tr)*tr
            if tend in bis_t: 
                y_out.append([int(subid), tend, bis[bis_t==tend][0]])
                x_out.append(x)
        return np.array(x_out), np.array(y_out)
    
    def process(self, mat_file):

        eeg, bis = self.get_eeg_bis_from_mat(mat_file)
        eeg_t, bis_t = self.get_eeg_bis_time(eeg, bis)
        print("0. data : ", eeg.shape, eeg_t.shape, bis.shape, bis_t.shape)
        
        x_data, t_data= self.cut_eeg_evenly(eeg, eeg_t)
        print("1. cut data evenly: ", x_data.shape)
        
        x_data = self.process_eeg(x_data, t_data)
        print("2. eeg filter and EMD: ", x_data.shape)
        
        x_data = self.process_norm(x_data)
        print("3. Normalize : ", x_data.shape)
        
        x_data, y_data = self.process_label( mat_file
                                           , bis, bis_t
                                           , x_data, t_data) 
        
        dset = np.concatenate((x_data, y_data), axis=1)
        
        
        return dset 
   
    def rnd_choice(self, dset, mask, n, rs):
        dset_target, dset_other = dset[mask], dset[mask==False]
        dset_target, _ = train_test_split( dset_target, train_size=n
                                         , shuffle=True, random_state=rs)
        dset =  np.concatenate((dset_target, dset_other))
        return dset 

def main():

    STEP_SEC = 2
    WINDOW_SEC = 6
    dp = DataProcess(STEP_SEC, WINDOW_SEC)
    
    mat_list = glob.glob('../data/org/case*.mat')
    
    # process 
    dset=[]
    for mat in mat_list:
        d = dp.process(mat)
        dset.extend(d)
        print(mat, d.shape, len(dset))
        print("===========================")
    dset = np.array(dset) 

    # data number selection
    mask = (dset[:,-1]>=30) & (dset[:,-1]<40) 
    dset = dp.rnd_choice(dset, mask, 10000, rs=34)
    mask = (dset[:,-1]>=40) & (dset[:,-1]<50) 
    dset = dp.rnd_choice(dset, mask, 10000, rs=34)
 
    # save 
    print("* Save ", dset.shape)
    np.save("results_data/s2_w5/dset.npy", dset)

if __name__ == '__main__':
    main()
