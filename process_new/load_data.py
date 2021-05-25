import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
  
class LoadDataset():
    def __init__(self, dset_path, xlen ):
        self.dset_path = dset_path
        self.xlen = xlen

    def plot_x(self, x, size=(16,4)):
        plt.figure(figsize=size)
        plt.plot(x, marker = 'o', alpha=0.4)
        plt.title("%s: %s"%(y_data[idx][0],y))
        plt.show()
    
    def hist_y(self, y, bins=10):
        plt.hist(y, bins=bins, alpha=0.4)
        plt.show()
    
    def subid_for_test(self, dset, subid):
        dset_subid = dset[:,-3]
        dset_other, dset_target = dset[dset_subid!=subid], dset[dset_subid==subid]
    
        return dset_other, dset_target    
    
    def train_valid_test_split(self, dset, size, rs):
        dset, dset_t = train_test_split( dset, test_size=size
                                       , shuffle=True, random_state=rs)
        dset, dset_v = train_test_split(dset, test_size=size
                                       , shuffle=True, random_state=rs)
    
        return dset, dset_v, dset_t
    
    def process_for_unsuper(self, size=0.2, rs = 34):
        dset = np.load(self.dset_path)
        print(dset.shape)
        
        dset, dset_r = self.subid_for_test(dset, 1)
        dset, dset_v, dset_t = self.train_valid_test_split(dset, size, rs)
        
        x_list, y_list = [], []
        for d in [dset, dset_v, dset_t, dset_r]:
            x, y = d[:,:self.xlen], d[:,self.xlen:]      
            x_list.append(x)
            y_list.append(y)
    
        return x_list, y_list 
    
