from drawTools import *

import os, sys, glob
import numpy as np
import random
from sklearn.preprocessing import StandardScaler



class DataProcess():
    def __init__(self, dataDir, test_idx, target_label):
        
        # load data 
        self.xset = np.load('%s/xset.npy'%dataDir)
        self.yset = np.load('%s/yset.npy'%dataDir)
        self.idx = np.load('%s/idx.npy'%dataDir)
        
        self.test_idx = test_idx
        self.target_label = target_label
        
        self.n_features = self.xset.shape[1]
        self.idx_list = list(set(self.idx))
        self.label = self.labeling_y()
        
        print(self.xset.shape, self.yset.shape, self.n_features)

    def labeling_y(self):
        y = self.yset
        y[y<65]=0
        #y[(y>=40) & (y<65)]=1
        y[y>=65]=1
        return y
    
    def draw_y_hist(self, r=1, save=None):
        hist={}
        for idx in self.idx_list:
            hist[idx]=self.label[self.idx==idx]
        draw_multi_hist(hist, r, save=save)

    def draw_x_hist(self, x_data, save=None):
        fig = plt.figure(figsize=(9,3))
        plt.subplot(1,2,1)
        for i in range(self.n_features):
            x = self.xset[:,i,:]
            x = x.reshape(x.shape[0]*x.shape[1])
            plt.title('After scaled')
            plt.hist(x, bins=50, range=(-100, 100), alpha=0.5) 
        plt.subplot(1,2,2)
        for i in range(self.n_features):
            x = x_data[:,:,i]
            x = x.reshape(x.shape[0]*x.shape[1])
            plt.title('After scaled')
            plt.hist(x, bins=50, alpha=0.5) 
        if save!=None:
            plt.savefig(save)
        else: plt.show()

    def train_process(self):

        xtrain = self.xset[self.idx != self.test_idx]
        features = [xtrain[:,i] for i in range(self.n_features)]
        # x process
        x_data, x_scaler = [], []
        for f in features:
            # Remove outliers
            x = self.remove_overx(f)
            # Scale 
            x, scaler = self.scale_x(x)
            # Reshape
            x = x.reshape((xtrain.shape[0], xtrain.shape[2], 1))
            x_data.append(x)
            x_scaler.append(scaler)
        
        x_data = np.concatenate((x_data), axis=2)
        y_data = self.label[self.idx != self.test_idx]
        
        # x y balance
        #x_train, y_train = balance_data(x_train, y_train)
        return x_data, y_data, x_scaler
    
    
    def test_process(self, x_scaler):
        xtest = self.xset[self.idx == self.test_idx]
        features = [xtest[:,i] for i in range(self.n_features)]
        # x process
        x_data = []    
        for i, f in enumerate(features):
            # Remove outliers
            x = self.remove_overx(f)
            # Scale 
            x = x_scaler[i].transform(x)        
            # Reshape
            x = x.reshape((xtest.shape[0], xtest.shape[2], 1))
            x_data.append(x)
        
        x_data = np.concatenate((x_data), axis=2)
        y_data = self.label[self.idx == self.test_idx]
        
        return x_data, y_data

    
    def remove_overx(self, x, limit=100):
        x1d = x.reshape(x.shape[0]*x.shape[1], 1)
        x1d[x1d>limit]=limit
        x1d[x1d<-limit]=-limit
    
        return x1d
    
    def scale_x(self, x, scaler=None):
        if scaler==None:
            scaler = StandardScaler()
            scaler.fit(x)
        x_scaled = scaler.transform(x)
        return x_scaled, scaler

    def balance_data(self, x_data, y_data, n=5000):

        label_list = list(set(y_data))
        print(label_list, 'target label is: ', self.target_label)

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




def main():
    dataDir = sys.argv[1]
    saveDir = sys.argv[2]
    if not os.path.isdir(saveDir): os.mkdir(saveDir)
    else: 
        print('! Already exist')
        exit()


    TEST_IDX = 19
    TARGET_LABEL = 1
    DP = DataProcess(dataDir, TEST_IDX, TARGET_LABEL)

    # start data process 
    x_data, y_data, x_scaler = DP.train_process()
    print('* Train process: ', x_data.shape, y_data.shape)
    
    x_train, y_train = DP.balance_data(x_data, y_data)
    print('* Balance trainset', x_train.shape, y_train.shape)
    
    x_test, y_test = DP.test_process(x_scaler)
    print('* Test process: ', x_test.shape, y_test.shape)

    # draw histograms
    DP.draw_y_hist(4,'%s/plot_idxlabel'%saveDir )
    DP.draw_x_hist(x_data, save='%s/plot_datadist'%saveDir)
    draw_multi_hist({'org': y_data, 'train': y_train, 'test': y_test},
                     save='%s/plot_datalabel'%saveDir)
    # save data 
    np.save('%s/x_data'%saveDir, x_train)
    np.save('%s/y_data'%saveDir, y_train)
    np.save('%s/x_test'%saveDir, x_test)
    np.save('%s/y_test'%saveDir, y_test)
    np.save('%s/x_scaler'%saveDir, x_scaler)


if __name__ == '__main__':
    main()
