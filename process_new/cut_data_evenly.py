import numpy as np
import h5py
import pandas as pd

class EEGProcess(): 
    def __init__( self, file_fix, eeg_window_size
                , bis_per_sec = 0.2, eeg_per_sec = 125):
        
        self.file_fix = file_fix
        self.eeg_window_size = eeg_window_size
        self.bis_per_sec = bis_per_sec
        self.eeg_per_sec = eeg_per_sec

    def _get_eeg_bis_from_mat(self, mat_file):
        f = h5py.File(mat_file, mode='r')
        data={}
        for a, b in f.items():
            data[a]=np.array(b)
        eeg = data['EEG'].ravel()
        bis = data['bis'].ravel()
        return eeg, bis
    
    def _get_time_arr(self, eeg, bis):
        eegt = np.array([(1/self.eeg_per_sec)*i for i in range(len(eeg))])
        bist = np.array([(1/self.bis_per_sec)*i for i in range(len(bis))])
        return eegt, bist
    
    def _split_by_time(self, x, t, mn_t, mx_t):
        mask = (t>=mn_t)&(t <= mx_t)
        x_cutted, t_cutted = x[mask], t[mask]
        return x_cutted, t_cutted
    
    def _split_and_filter_hclass( self, eeg, eegt, i, c
                                , step_size, abnor_cutval, abnor_cnt):
        
        n_step = int(np.trunc(len(eeg)/step_size))
        x_out, t_out, y_out = [], [], []
        for step in range(n_step):
            start = step*step_size
            end = start+self.eeg_window_size
            x = eeg[start:end]
            t = eegt[start:end]
            cnt = len(x[np.abs(x)<abnor_cutval])
            
            if (len(x)==(self.eeg_window_size)) and (cnt<abnor_cnt):
                x_out.append(x)
                t_out.append(t)
                y_out.append([i, c])
                
        return np.array(x_out), np.array(t_out), np.array(y_out)
    
    def process_high_class(self, exel_file, step_size, abnor_cutval=0.04, abnor_cnt=200):
        exel_file = pd.read_excel(exel_file)
    
        xh_data, th_data, yh_data = [], [], []
        for i, c, _, _, ts, te in exel_file.values:
            # get eeg and bis 
            mat_file = self.file_fix%i
            eeg, bis = self._get_eeg_bis_from_mat(mat_file)
            eegt, bist = self._get_time_arr(eeg, bis)
            
            # split eeg, bis by time step
            eeg, eegt = self._split_by_time(eeg, eegt, ts, te)
            bis, bist = self._split_by_time(bis, bist, ts, te)
            
            # split regular step
            xh, th, yh = self._split_and_filter_hclass( eeg, eegt
                                                      , i, c, step_size
                                                      , abnor_cutval, abnor_cnt)        
        
            xh_data.extend(xh)
            th_data.extend(th)
            yh_data.extend(yh)
        
        return np.array(xh_data), np.array(th_data), np.array(yh_data)
    
    def _split_and_filter_lclass( self, eeg, eegt, bis, bist, i
                                , abnor_cutval, abnor_cnt):

        x_out, t_out, y_out = [], [], []
        for b, bt in zip(bis, bist):
            start = bt-(self.eeg_window_size/self.eeg_per_sec)
            end = bt
            mask = (eegt>=start) & (eegt<end)
            x = eeg[mask]
            t = eegt[mask]
            cnt = len(x[np.abs(x)<abnor_cutval])
            if (len(x)==(self.eeg_window_size)) and (cnt<abnor_cnt):
                x_out.append(x)
                t_out.append(t)
                y_out.append([i, b])
        
        return np.array(x_out), np.array(t_out), np.array(y_out) 
    
    def process_low_class(self, exel_file, abnor_cutval=0.04, abnor_cnt=200):

        exel_file = pd.read_excel(exel_file)

        xl_data, tl_data, yl_data = [], [], []
        for i, _, _, ts, te in exel_file.values:
            mat_file = self.file_fix%i
            eeg, bis = self._get_eeg_bis_from_mat(mat_file)
            eegt, bist = self._get_time_arr(eeg, bis)
            
            # split eeg, bis by time step
            eeg, eegt = self._split_by_time(eeg, eegt, ts, te)
            bis, bist = self._split_by_time(bis, bist, ts, te)
            
            # split regular step
            xl, tl, yl = self._split_and_filter_lclass( eeg, eegt, bis, bist, i
                                                 , abnor_cutval, abnor_cnt)
            xl_data.extend(xl)
            tl_data.extend(tl)
            yl_data.extend(yl)
        
        return np.array(xl_data), np.array(tl_data), np.array(yl_data)


