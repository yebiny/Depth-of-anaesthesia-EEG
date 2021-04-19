import numpy as np
import sys, os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_dataset(data_path, tag='train'):
    x = np.load('%s/x_%s.npy'%(data_path,tag))
    y = np.load('%s/y_%s.npy'%(data_path,tag))
    l = np.load('%s/l_%s.npy'%(data_path,tag))
    return x, y , l

def save_dataset(x,y,l, save_path, tag='train'):
    np.save('%s/x_%s.npy'%(save_path,tag), x)
    np.save('%s/y_%s.npy'%(save_path,tag), y)
    np.save('%s/l_%s.npy'%(save_path,tag), l)

def relabel(l_data):
    l_data[l_data<=1]=0
    l_data[l_data>=2]=1

    return l_data

def balance_data(x_data, y_data, l_data,  size):

    x0, y0, l0 = x_data[l_data==0], y_data[l_data==0], l_data[l_data==0]
    x1, y1, l1 = x_data[l_data==1], y_data[l_data==1], l_data[l_data==1]

    x, _, y, _ = train_test_split(x0, y0, train_size=size, shuffle=True, random_state=11)
    x, _, l, _ = train_test_split(x0, l0, train_size=size, shuffle=True, random_state=11)

    x =  np.concatenate((x, x1))
    y =  np.concatenate((y, y1))
    l =  np.concatenate((l, l1))

    return x, y, l

def draw_hist(train, valid, bins, save=None):
    fig = plt.figure(figsize=(8, 3))
    plt.subplot(1,2,1)
    plt.title('train')
    plt.hist(train, color = '#9467bd', rwidth=0.9, bins=bins, alpha=0.6)
    plt.subplot(1,2,2)
    plt.title('valid')
    plt.hist(valid, color = '#9467bd', rwidth=0.9, bins=bins, alpha=0.6)
    if save: plt.savefig(save)
    else: plt.show()
    plt.close()

def main():
    data_path=sys.argv[1]
    save_path = '%s/%s'%(data_path, sys.argv[2])
    
    if not os.path.isdir(save_path): os.mkdir(save_path)
    else: exit()
    
    sys.stdout = open('%s/log.txt'%save_path,'w')
    
    b_ratio=0.2
    print('b_ratio: ',b_ratio)

    x_train, y_train, l_train = get_dataset(data_path, tag='train')
    x_valid, y_valid, l_valid = get_dataset(data_path, tag='valid')
    print('data: ', x_train.shape, x_valid.shape)
    
    l_train = relabel(l_train)
    l_valid = relabel(l_valid)
    
    x_train, y_train, l_train = balance_data(x_train, y_train, l_train, b_ratio)
    x_valid, y_valid, l_valid = balance_data(x_valid, y_valid, l_valid, b_ratio)
    print('balanced data: ', x_train.shape, x_valid.shape)
   
    draw_hist(y_train, y_valid, 20, save = save_path+'/yhist')
    draw_hist(l_train, l_valid, 2, save = save_path+'/lhist')
    
    save_dataset(x_train, y_train, l_train, save_path, tag='train')
    save_dataset(x_valid, y_valid, l_valid, save_path, tag='valid')


if __name__ == '__main__':
    main()
