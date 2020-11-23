import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

def data_scaler(x_data, limit = 100):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    features = [x_data[:,i] for i in range(x_data.shape[1])]
    x_scaler = []
    x_out = []
    
    for f in features:
        
        # remove over limit values
        f[f>limit]=limit
        f[f<-limit]=-limit
        f_1d = f=f.reshape(f.shape[0]*f.shape[1], 1)

        # data scale-standard deviation
        scaler = StandardScaler()
        scaler.fit(f_1d)
        f_scaled = scaler.transform(f_1d)
        
        # draw data befor & afeter sacaling
        ax1.hist(f_1d, bins=50, alpha=0.5)
        ax2.hist(f_scaled, bins=50, alpha=0.5)
        
        # return removed & scaled data and scaler 
        x_out.append(f_scaled.reshape(len(x_data), 3750, 1))
        x_scaler.append(scaler)
        
    plt.show()

    x_out = np.concatenate((x_out[0], x_out[1]), axis=2)
    print(x_data.shape, x_out.shape)
    return x_out, x_scaler


def balance_data(xset, yset, test_size=0.1, n=2500):
    print('* Input data sahpe: ', xset.shape, yset.shape)
    
    
    y_labeled=[]
    for y in yset:
        if y >=65: out = 2
        #if (y>=65) & (y<85): out= 2
        elif (y>=40) & (y<65): out= 1
        else: out=0
        y_labeled.append(out)
    y_labeled = np.array(y_labeled)
    

    # testset
    x_data, x_test, y_data, y_test = train_test_split(xset, y_labeled, test_size=test_size, random_state=42)
    print('* split test data: ', x_data.shape, x_test.shape, y_data.shape, y_test.shape)

    
    # trainset
    x0 = x_data[y_data==0]
    x1 = x_data[y_data==1]
    x2 = x_data[y_data==2]
    print('* trainset each label number: ', x0.shape, x1.shape, x2.shape)
    
    x0 = np.array(random.sample(list(x0), n))
    x1 = np.array(random.sample(list(x1), n))
    
    x_data = np.concatenate((x0, x1, x2), axis=0)
    y_data = np.array([0 for i in range(n)]+[1 for i in range(n)]+[2 for i in range(len(x2))])
    print('* reduce trainset to sampling',  x_data.shape, y_data.shape)
    
    # validset
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    print('* split trainset and validset', x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

        
    plt.figure(figsize=(12,3))
    plt.subplot(1,4,1)
    plt.title('origin')
    plt.hist(y_labeled)  
          
    plt.subplot(1,4,2)
    plt.title('test')
    plt.hist(y_test)
          
    plt.subplot(1,4,3)
    plt.title('train')
    plt.hist(y_train)
          
    plt.subplot(1,4,4)
    plt.title('valid')
    plt.hist(y_valid)
    
    
    print('* final output shape: ', x_train.shape, x_valid.shape, x_test.shape)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def main():
    eegPreDir = arvs[0]
    xset = np.load('%s/xset.npy'%eegPreDir)
    yset = np.load('%s/yset.npy'%eegPreDir)

    x_scaled, x_scaler = data_scaler(xset)
    x_train, x_valid, x_test, y_train, y_valid, y_test = balance_data(x_scaled, yset, n=6000)

if __name__=='__main__':
        main()
