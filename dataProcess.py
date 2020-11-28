import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler


def x_flatten(x):
    x = x.reshape(x.shape[0]*x.shape[1], 1)
    return x

def draw_x_dist(x_data, save=None):
    fig = plt.figure(figsize=(6,4))
    for i in range(2):
        x = x_flatten(x_data[:,:,i])
        plt.hist(x, bins=50, alpha=0.4)
    plt.legend(['ch_1','ch_2'])
    if save!=None:
        plt.savefig(save)
    else: plt.show()

def remove_overval(x, limit=100):
    x1d = x_flatten(x)
    x1d[x1d>limit]=limit
    x1d[x1d<-limit]=-limit

    return x1d

def scale_x(x, scaler=None):
    if scaler==None:
        scaler = StandardScaler()
        scaler.fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled, scaler


def labeling_y(yset):

    yset[yset<40]=0
    yset[(yset>=40) & (yset<65)]=1
    yset[yset>=65]=2

    return yset

def balance_data(x_data, y_data, test_size=0.1, n=5000):
    r = int(max(y_data))
    # trainset
    x_split=[]
    x2 = x_data[y_data==2]
    for i in range(r):
        x = x_data[y_data==i]
        x_balance = np.array(random.sample(list(x), n))
        x_split.append(x_balance)
        print('label %i '%i, x.shape, x_balance.shape)
    print('label 2 :', x2.shape)

    # Final output
    x_train = np.concatenate((x_split[0], x_split[1], x2), axis=0)
    y_train = np.array([0 for i in range(n)]+[1 for i in range(n)]+[2 for i in range(len(x2))] )

    return x_train, y_train


def trian_process(xset, yset):
    x_data = []
    x_scaler=[]
    print('* input shape: ', xset.shape)
    features = [xset[:,i] for i in range(xset.shape[1])]
    
    for f in features:
        # Remove outliers
        x = remove_overval(f)
        # Scale 
        x, scaler = scale_x(x)
        # Reshape
        x = x.reshape((xset.shape[0], 3750, 1))

        x_data.append(x)
        x_scaler.append(scaler)

    x_data = np.concatenate((x_data[0], x_data[1]), axis=2)
    y_data = labeling_y(yset)
    
    x_train, y_train = balance_data(x_data, y_data)
    print(x_train.shape, y_train.shape)
    
    return x_train, y_train, x_scaler


def test_process(xset, yset, x_scaler):
    print(xset.shape, yset.shape)
    x_data = []    
    features = [xset[:,i] for i in range(xset.shape[1])]
    for i, f in enumerate(features):
        # Remove outliers
        x = remove_overval(f)
        # Scale 
        x = x_scaler[i].transform(x)        
        # Reshape
        x = x.reshape((xset.shape[0], 3750, 1))
        x_data.append(x)

    x_test = np.concatenate((x_data[0], x_data[1]), axis=2)
    y_test = labeling_y(yset)
    print(x_test.shape, y_test.shape)
    
    return x_test, y_test


def draw_multihist(plots, w, save=None):

    h = len(plots)/w
    plt.figure(figsize=(w*3,h*3))
    for i, p in enumerate(plots):
        plt.subplot(h,w,i+1)
        plt.title(p)
        plt.hist(plots[p], facecolor='#9467bd', bins=3, rwidth=0.9, alpha=0.6)  
          
    if save!=None:
        plt.savefig(save) 
    else: plt.show()


def main():
    # options
    eegDir = sys.argv[1]
    nDir = len(glob.glob('0-data/res_process/eeg1/data*'))   
    outDir= '%s/data%i'%(eegDir,nDir)
    
    # save log
    if not os.path.isdir(outDir): os.mkdir(outDir)
    sys.stdout = open('%s/log.txt'%outDir,'w')
    print('input: %s, output: %s'%(eegDir, outDir))
   
    # process
    xset = np.load('%s/xset.npy'%eegDir)
    yset = np.load('%s/yset.npy'%eegDir)
    ntest = len(xset)-1480
    print(xset.shape, yset.shape)
    
    x_data, y_data, x_scaler = trian_process(xset[:ntest], yset[:ntest] )
    x_test, y_test = test_process(xset[ntest:], yset[ntest:], x_scaler)
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=99)
  
    # save process results
    np.save('%s/x_data'%outDir, x_data)
    np.save('%s/y_data'%outDir, y_data)
    np.save('%s/x_test'%outDir, x_test)
    np.save('%s/y_test'%outDir, y_test)
    np.save('%s/x_scaler'%outDir, x_scaler)

    # draw images
    draw_multihist({ 'org':y_data,
                    'train':y_train,
                    'valid':y_valid,
                    'test': y_test}, 
                    w=4, save='%s/plot_classes'%outDir)
    
    draw_x_dist(x_data, save='%s/plot_trainDist'%outDir)
    draw_x_dist(x_test, save='%s/plot_testDist'%outDir)

if __name__=='__main__':
    main()
