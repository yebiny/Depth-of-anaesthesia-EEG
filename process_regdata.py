import numpy as np
import sys, os

def get_dataset(data_path, tag='train'):
    x = np.load('%s/x_%s.npy'%(data_path,tag))
    y = np.load('%s/y_%s.npy'%(data_path,tag))
    l = np.load('%s/l_%s.npy'%(data_path,tag))
    return x, y , l

def save_dataset(x,y,l, save_path, tag='train'):
    np.save('%s/x_%s.npy'%(save_path,tag), x)
    np.save('%s/y_%s.npy'%(save_path,tag), y)
    np.save('%s/l_%s.npy'%(save_path,tag), l)

def set_condition(x_data, y_data, l_data, idx, val=None):
    if val:
        x_data, y_data, l_data = x_data[l_data<idx], y_data[l_data<idx], l_data[l_data<idx]
        x_data, y_data, l_data = x_data[y_data>val], y_data[y_data>val], l_data[y_data>val]
    else:
        x_data, y_data, l_data = x_data[l_data>=idx], y_data[l_data>=idx], l_data[l_data>=idx]
    return x_data, y_data, l_data

def draw_l_hist(y_data, save=None):
    fig = plt.figure(figsize=(4, 3))
    plt.hist(y_data, color = '#9467bd', rwidth=0.9, bins=5, alpha=0.6)
    if save: plt.savefig(save)
    else: plt.show()
    plt.close()

def normalize_y(data, mn, mx):
    data = (data-mn)/(mx-mn)
    return data

def main():
    data_path=sys.argv[1]
    save_path=sys.argv[2]
    
    if not os.path.isdir(save_path): os.mkdir(save_path)
    else: exit()
    
    sys.stdout = open('%s/log.txt'%save_path,'w')
    
    idx= 2
    mn = 10
    mx = 65
    val = 100
    print('idx, mn, mx, val')
    print(idx, mn, mx, val)
    
    x_train, y_train, l_train = get_dataset(data_path, tag='train')
    x_valid, y_valid, l_valid = get_dataset(data_path, tag='valid')
    print(x_train.shape, x_valid.shape)
    x_train, y_train, l_train = set_condition(x_train, y_train, l_train, idx, val)
    x_valid, y_valid, l_valid = set_condition(x_valid, y_valid, l_valid, idx, val)
    print(x_train.shape, x_valid.shape)
   
    y_train = normalize_y(y_train, mn, mx)
    y_valid = normalize_y(y_valid, mn, mx)
    
    draw_y_hist(y_train, save_path+'/yhist')
    save_dataset(x_train, y_train, l_train, save_path, tag='train')
    save_dataset(x_valid, y_valid, l_valid, save_path, tag='valid')


if __name__ == '__main__':
    main()
