import numpy as np
from sklearn.model_selection import train_test_split

def get_ch(x_imfs, ch):
    x_ch = x_imfs[:,ch,:]
    x_ch = np.expand_dims(x_ch, 2)
    x_ch = (x_ch-np.mean(x_ch))/np.std(x_ch)
    
    return x_ch

def get_ch_add(x_imfs):
    x_ch0 = get_ch(x_imfs, 0)
    x_ch1 = get_ch(x_imfs, 1)
    x_ch_add = (x_ch0+x_ch1)/2
    
    return x_ch_add

def get_ch_concat(x_imfs):
    x_ch0 = get_ch(x_imfs, 0)
    x_ch1 = get_ch(x_imfs, 1)
    x_ch_concat = np.concatenate((x_ch0, x_ch1), axis=2)

    return x_ch_concat

def generate_datasets(x_path, y_path, mask_path, ch, valid_size=0.15):
    x_data = np.load(x_path) 
    y_data = np.load(y_path)[:,-1]
    mask = np.load(mask_path)
    print('* load dataset: ', x_data.shape, y_data.shape)
    
    x_data, y_data = x_data[mask], y_data[mask]
    print('* masked dataset: ', x_data.shape, y_data.shape)
    
    if type(ch)==int:
        x_data = get_ch(x_data, ch)
    else:
        if ch=='add':
            x_data=get_ch_add(x_data)
        elif ch=='concat':
            x_data=get_ch_concat(x_data)
        else:
            print('channel not valid')
    print('* final dataset: ', x_data.shape, y_data.shape)
    
    return x_data, y_data 
