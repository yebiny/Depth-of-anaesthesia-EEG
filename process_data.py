from draw import *
import copy
import os, sys, glob
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

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
    
    def y_process(self, y_range):
        y_data = np.clip(self.yset, 20, 100)
        y_data = y_data/10
        y_data = np.log10(y_data)
        
        y_label = np.clip(self.yset, 20, 100)
        for ri, r in enumerate(y_range[:-1]):
            print(r, ri)
            mask = (y_label>r)&(y_label<=y_range[ri+1])
            y_label[mask]= ri

        return y_data, y_label


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

        # x process
        x_data = []    
        for i, f in enumerate(features):
            # Remove outliers
            x = self._remove_overx(f)
            # Scale 
            x = x_scaler[i].transform(x)        
            # Reshape
            x = x.reshape((xtest.shape[0], xtest.shape[2], 1))
            x_data.append(x)
        
        x_data = np.concatenate((x_data), axis=2)
        y_data = self.label[self.idx == self.test_idx]
        
        return x_data, y_data

    

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

    def train_test_split(self, x_data, y_data, y_label, test_target):
        x_train, x_test = x_data[self.idx!=test_target], x_data[self.idx==test_target]
        y_train, y_test = y_data[self.idx!=test_target], y_data[self.idx==test_target]
        y_train_label, y_test_label = y_label[self.idx!=test_target], y_label[self.idx==test_target]
        
        return x_train, y_train, y_train_label, x_test, y_test, y_test_label

def main():
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    y_range = [0,40,65,85,100]
    test_target=19
    if not os.path.isdir(save_path): os.mkdir(save_path)
    else: 
        print('! %s Already exist'%save_path)
        exit()

    dp = DataProcess(data_path)
    
    # start data process 
    x_data, x_scaler = dp.x_process()
    y_data, y_label = dp.y_process(y_range)
    print(x_data.shape, y_data.shape)
    
    x_train, y_train, y_train_label, x_test, y_test, y_test_label = dp.train_test_split( x_data
                                                                                       , y_data
                                                                                       , y_label
                                                                                       , test_target)
    #if mode=='category':
    #    TEST_IDX = 19
    #    TARGET_LABEL = 1
    #    x_train, y_train = DP.balance_data(x_data, y_data)
    #    print('* Balance trainset', x_train.shape, y_train.shape)
    #    
    #    x_test, y_test = DP.test_process(x_scaler)
    #    print('* Test process: ', x_test.shape, y_test.shape)

    #    # draw histograms
    #    DP.draw_y_hist(4,'%s/plot_idxlabel'%saveDir )
    #    DP.draw_x_hist(x_data, save='%s/plot_datadist'%saveDir)
    #    draw_multi_hist({'org': y_data, 'train': y_train, 'test': y_test},
    #                     save='%s/plot_datalabel'%saveDir)
    #
    #    np.save('%s/x_train'%saveDir, x_train)
    #    np.save('%s/y_train'%saveDir, y_train)
    #    np.save('%s/x_test'%saveDir, x_test)
    #    np.save('%s/y_test'%saveDir, y_test)
    #
    # save data 
    np.save('%s/x_train'%save_path, x_train)
    np.save('%s/y_train'%save_path, y_train)
    np.save('%s/y_train_label'%save_path, y_train_label)
    np.save('%s/x_test'%save_path, x_test)
    np.save('%s/y_test'%save_path, y_test)
    np.save('%s/y_test_label'%save_path, y_test_label)
    np.save('%s/x_scaler'%save_path, x_scaler)

if __name__ == '__main__':
    main()
