import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DataProcess():
    def __init__(self, dset_path, xlen):
        self.dset_path = dset_path
        self.xlen = xlen
        self.dset = np.load(dset_path)
        print(self.dset.shape)

    def plot_x(self, x, size=(16,4)):
        plt.figure(figsize=size)
        plt.plot(x, marker = 'o', alpha=0.4)
        plt.show()
    
    def hist_with_number(self,x,save_path=None, title=None):
        classes = set(x)
        
        figure = plt.figure(figsize=(8,6))
        plt.hist( x
                , range=(min(classes)-0.5,max(classes)+0.5)
                , bins=len(classes)
                , alpha=0.4
                , rwidth=0.9)
        plt.xticks(list(classes))
    
        n_classes=[]
        for c in classes:
            n_class = len(x[x==c])
            n_classes.append(n_class)
            plt.text(c,n_class,n_class,                 
                     color='slateblue',
                     fontweight='bold',
                     horizontalalignment='center',  
                     verticalalignment='bottom')
        if title: plt.title(title)
        if save_path: plt.savefig(save_path)
        else: plt.show()
        
        return n_classes

    def subid_for_test(self, dset, subid):
        dset_subid = dset[:,self.xlen]
        dset_other, dset_target = dset[dset_subid!=subid], dset[dset_subid==subid]

        return dset_other, dset_target

    def rnd_choice(self, dset, mask, n, rs):
        dset_target, dset_other = dset[mask], dset[mask==False]
        dset_target, _ = train_test_split( dset_target, train_size=n
                                         , shuffle=True, random_state=rs)
        dset =  np.concatenate((dset_target, dset_other))
        return dset

    def class_labeling(self):
        dset_labeled=[]
        for d in self.dset:
            if d[-1]>20 and d[-1]<40:
                d = np.append(d, 0)
            elif d[-1]>=40 and d[-1]<65:
                d = np.append(d, 1)
            elif d[-1]>=65 and d[-1]<85:
                d = np.append(d, 2)
            elif d[-1]>=85:
                d = np.append(d, 3)
            else:
                d = np.append(d, -1)
            dset_labeled.append(d)
        return np.array(dset_labeled)



def main():
    
    SAVE_PATH = 'results_data/s1_w6'
    dp = DataProcess('%s/eegset.npy'%SAVE_PATH, 6*125)
    
    dset = dp.class_labeling()
    dset = dset[dset[:,-1]!=-1]
    dset, dset_real = dp.subid_for_test(dset, 5)
    n_classes = dp.hist_with_number(dset[:,-1], SAVE_PATH+'/hist_bfrnd', title='dataset before rnd')  
    _ = dp.hist_with_number(dset_real[:,-1], SAVE_PATH+'/hist_real', title='realset')  
     
    for i in range(2):
        mask = dset[:,-1]==i
        dset = dp.rnd_choice(dset, mask, n=np.min(n_classes), rs=34)
    dset = np.array(dset, dtype='float32')
    dset_real = np.array(dset_real, dtype='float32')
    print(dset.shape, dset_real.shape)
    
    dp.hist_with_number(dset[:,-1], SAVE_PATH+'/hist_rnd', title='dataset after rnd')  
    np.save("%s/dset.npy"%SAVE_PATH, dset)
    np.save("%s/dset_real.npy"%SAVE_PATH, dset_real)

if __name__ == '__main__':
    main()
