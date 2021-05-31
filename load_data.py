import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
  
class LoadDataset():
    def __init__(self, dset_path, xlen ):
        self.dset_path = dset_path
        self.xlen = xlen

    def train_valid_test_split(self, dset, vsize, tsize, rs):
        dset, dset_v = train_test_split( dset, test_size=vsize
                                       , shuffle=True, random_state=rs)
        dset_v, dset_t = train_test_split(dset_v, test_size=tsize, shuffle=False)
    
        return dset, dset_v, dset_t
    
    def process_for_cfy(self, vsize=0.2, tsize=0.1, rs = 34):
        dset = np.load(self.dset_path)
        dset_r = np.load(self.dset_path)

        dset, dset_v, dset_t = self.train_valid_test_split(dset, vsize,tsize, rs)
        
        x_list, y_list = [], []
        for d in [dset, dset_v, dset_t, dset_r]:
            x, y = d[:,:self.xlen], d[:,-1]      
            x = np.expand_dims(x, 2)
            x = np.array(x, dtype='float32')
            y = np.array(y, dtype='int')
            x_list.append(x)
            y_list.append(y)
    
        return x_list, y_list 
