import os, sys, glob, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

class DataProcess():
    def __init__(self, data_path):
        
        # load data 
        self.xset = np.load('%s/xset.npy'%data_path)
        self.yset = np.load('%s/yset.npy'%data_path)
        self.idx = np.load('%s/idx.npy'%data_path)
        
        self.n_features = self.xset.shape[1]
        self.idx_list = list(set(self.idx))
        print(self.xset.shape, self.yset.shape, self.n_features)
    
    def _remove_overx(self, x, limit=100):
        x1d = x.reshape(x.shape[0]*x.shape[1], 1)
        x1d[x1d>limit]=limit
        x1d[x1d<-limit]=-limit
    
        return x1d
    
    def _scale_x(self, x, scaler=None):
        if scaler==None:
            scaler = StandardScaler()
            scaler.fit(x)
        x_scaled = scaler.transform(x)
        return x_scaled, scaler
    
    def x_process(self):
        features = [self.xset[:,i] for i in range(self.n_features)]
        
        x_data, x_scaler = [], []
        for f in features:
            # Remove outliers
            x = self._remove_overx(f)
            # Scale 
            x, scaler = self._scale_x(x)
            # Reshape
            x = x.reshape((self.xset.shape[0], self.xset.shape[2], 1))
            x_data.append(x)
            x_scaler.append(scaler)
        x_data = np.concatenate((x_data), axis=2)
        
        return x_data, x_scaler
    
    def y_process(self, y_range, scaler):
        self.n_bins=len(y_range)-1
        if scaler == 'divide':
            y_data = np.clip(self.yset, 0, 100)
            y_data = y_data*0.01

        l_data = np.clip(self.yset, 10, 100)
        for ri, r in enumerate(y_range[:-1]):
            mask = (l_data>r)&(l_data<=y_range[ri+1])
            l_data[mask]= ri

        return y_data, l_data


    def draw_x_hist(self, x_data, save=None):
        fig = plt.figure(figsize=(9,3))
        plt.subplot(1,2,1)
        for i in range(self.n_features):
            x = self.xset[:,i,:]
            x = x.reshape(x.shape[0]*x.shape[1])
            plt.title('Before scaled')
            plt.hist(x, bins=50, range=(-100, 100), alpha=0.35) 
        plt.subplot(1,2,2)
        for i in range(self.n_features):
            x = x_data[:,:,i]
            x = x.reshape(x.shape[0]*x.shape[1])
            plt.title('After scaled')
            plt.hist(x, bins=50, alpha=0.35) 
        if save!=None:
            plt.savefig(save)
        else: plt.show()
        plt.close()
    
    def draw_y_hist(self, y_data, save=None):
        fig = plt.figure(figsize=(9,3))
        plt.subplot(1,2,1)
        plt.hist(self.yset, bins=20, color = '#9467bd', alpha=0.6)
        plt.subplot(1,2,2)
        plt.hist(y_data, bins=20, color = '#9467bd', alpha=0.6)
        if save: plt.savefig(save)
        else: plt.show()
        plt.close()

    def draw_l_hist(self, l_data, save=None):
       
        fig = plt.figure(figsize=(4, 3))
        plt.hist(l_data, color = '#9467bd', rwidth=0.9, bins=5, alpha=0.6)
        if save: plt.savefig(save)
        else: plt.show()
        
        fig = plt.figure(figsize=(18, 12))
        for idx in self.idx_list:
            data=l_data[self.idx==idx]
            plt.subplot(6, 6, idx)
            plt.title(idx)
            plt.hist(data, color = '#9467bd', rwidth=0.9, bins=self.n_bins, alpha=0.6)
        if save: plt.savefig(save+"_idx")
        else: plt.show()
        plt.close()

    def balance_data(self, x_data, y_data, n=5000):

        label_list = list(set(y_data))
        #print(label_list, 'target label is: ', self.target_label)

        x_out, y_out=[], []
        for label in label_list:
            if label==self.target_label:
                x_balance = x_data[y_data==self.target_label]
                y_balance = [label for i in range(len(x_balance))]
            else:
                x_balance = x_data[y_data==label]
                x_balance = np.array(random.sample(list(x_balance), n))
                y_balance = [label for i in range(n)]
            x_out.append(x_balance)
            y_out.append(y_balance)

        # Final output
        x_data = np.concatenate((x_out), axis=0)
        y_data = np.concatenate((y_out), axis=0)

        return x_data, y_data

    def train_test_split(self, x_data, y_data, l_data, test_target):
        x_train, x_test = x_data[self.idx!=test_target], x_data[self.idx==test_target]
        y_train, y_test = y_data[self.idx!=test_target], y_data[self.idx==test_target]
        y_train_label, y_test_label = l_data[self.idx!=test_target], l_data[self.idx==test_target]
        
        return x_train, y_train, y_train_label, x_test, y_test, y_test_label
    
    def train_valid_split(self, x_data, y_data, l_data, size, rs):
        x, xv, y, yv = train_test_split(x_data, y_data, test_size=0.2, shuffle=True, random_state=rs)
        _, _, l, lv = train_test_split(x_data, l_data, test_size=0.2, shuffle=True, random_state=rs)

        return x, y, l, xv, yv, lv

def main():
    data_path = sys.argv[1]
    save_path = '%s/%s'%(data_path, sys.argv[2])
    if not os.path.isdir(save_path): os.mkdir(save_path)
    else: 
        print('! %s Already exist'%save_path)
        exit()
    
    opt={ 
    'Y_RANGE': [0, 40, 60, 80, 100], 
    #y_range = [0,21,41,61,78,100]
    'Y_SCALE': 'divide',
    'TEST_TARGET': 14,
    'VALID_SIZE': 0.2,
    'RS': 34,
    }
    sys.stdout = open('%s/log.txt'%save_path,'w') 
    print(opt)
    
    dp = DataProcess(data_path)
    
    # start data process 
    x_data, x_scaler = dp.x_process()
    y_data, l_data = dp.y_process(opt['Y_RANGE'], opt['Y_SCALE'])
    print('* total data: ', x_data.shape, y_data.shape, l_data.shape)
    
    # draw data plots
    dp.draw_x_hist(x_data, '%s/xhist'%save_path)
    dp.draw_y_hist(y_data, '%s/yhist'%save_path)
    dp.draw_l_hist(l_data, '%s/lhist'%save_path)
   
    # test data split
    x_data, y_data, l_data, x_test, y_test, l_test = dp.train_test_split( x_data
                                                                        , y_data
                                                                        , l_data
                                                                        , opt['TEST_TARGET'])
    
    x_train, y_train, l_train, x_valid, y_valid, l_valid = dp.train_valid_split( x_data
                                                                                , y_data
                                                                                , l_data
                                                                                , opt['VALID_SIZE']
                                                                                , opt['RS'])

    print('* trainset: ', x_train.shape, y_train.shape, l_test.shape)
    print('* vlaidset: ', x_valid.shape, y_valid.shape, l_test.shape)
    print('* testset : ', x_test.shape, y_test.shape, l_test.shape)

    # save data 
    np.save('%s/x_train'%save_path, x_train)
    np.save('%s/y_train'%save_path, y_train)
    np.save('%s/l_train'%save_path, l_train)
    np.save('%s/x_valid'%save_path, x_valid)
    np.save('%s/y_valid'%save_path, y_valid)
    np.save('%s/l_valid'%save_path, l_valid)
    np.save('%s/x_test'%save_path, x_test)
    np.save('%s/y_test'%save_path, y_test)
    np.save('%s/l_test'%save_path, l_test)
    np.save('%s/x_scaler'%save_path, x_scaler)

if __name__ == '__main__':
    main()
