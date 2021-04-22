import os, sys, glob, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from draw import draw_lhist

def save_data(x, y, l, save_path, tag): 
    x0, x1, x2 = x[:,0], x[:,1], x[:,2]
    x = np.concatenate((np.expand_dims(x0, 2), np.expand_dims(x1, 2), np.expand_dims(x2, 2)), axis=2)
    np.save('%s/x_%s.npy'%(save_path,tag), x)
    np.save('%s/x0_%s.npy'%(save_path,tag), x0)
    np.save('%s/x1_%s.npy'%(save_path,tag), x1)
    np.save('%s/x2_%s.npy'%(save_path,tag), x2)
    np.save('%s/y_%s.npy'%(save_path,tag), y)
    np.save('%s/l_%s.npy'%(save_path,tag), l)

def load_data(data_path): 
    xset = np.load('%s/xset.npy'%data_path)
    yset = np.load('%s/yset.npy'%data_path)
    idx = np.load('%s/idx.npy'%data_path)
    return xset, yset, idx

def remove_idx(x, y, idx, val):
    x, y,idx = x[idx!=val], y[idx!=val], idx[idx!=val]
    return x, y, idx

def remove_abnorm_y(x, y, idx, val):
    x, y, idx = x[y>=val], y[y>=val], idx[y>=val]
    return x, y, idx

def clip_xval(x, clipx):
    x = np.clip(x, -clipx, clipx)
    return x

def mask_yval(x, y, idx, ymask):
    x, y, idx = x[ymask], y[ymask], idx[ymask]
    return x, y, idx

def get_label(y, tag):
    l=[]
    if tag=='class4':
        for val in y:
            if val<40: l.append(0)
            elif (val>=40)&(val<65): l.append(1)
            elif (val>=65)&(val<85): l.append(2)
            else: l.append(3)
    
    elif tag=='class2':
        for val in y:
            if val<65: l.append(0)
            else: l.append(1)

    elif tag=='upper':
        for val in y:
            if val<85: l.append(0)
            else: l.append(1)
    
    elif tag=='under':    
        for val in y:
            if val<40: l.append(0)
            else: l.append(1)

    else: print("Tag Error")
    return np.array(l)


def balance_data(x, y, l, size, val, rs=11):
    

    x0, y0, l0 = x[y<val], y[y<val], l[y<val]
    x1, y1, l1 = x[y>=val], y[y>=val], l[y>=val]

    x, _, y, _ = train_test_split(x0, y0, train_size=size, shuffle=True, random_state=rs)
    x, _, l, _ = train_test_split(x0, l0, train_size=size, shuffle=True, random_state=rs)

    x =  np.concatenate((x, x1))
    y =  np.concatenate((y, y1))
    l =  np.concatenate((l, l1))

    return x, y, l

def split_data(x, y, l, size, rs):
    _, _, y, yv = train_test_split(x, y, test_size=size, shuffle=True, random_state=rs)
    x, xv, l, lv = train_test_split(x, l, test_size=size, shuffle=True, random_state=rs)

    return x, y, l, xv, yv, lv

def norm_y(y, method):
    y = (y-method[0])/(method[1]-method[0]) 
    return y

def process(x, y, idx, opt):

    tag_cut = {'class4':65
              ,'class2':65   
              ,'upper' :85   
              ,'under' :40}  
    
    if 1 in opt.keys(): 
        x, y, idx = remove_idx(x,y,idx, opt[1])
    
    if 2 in opt.keys(): 
        x, y, idx = remove_abnorm_y(x,y,idx, opt[2])
    
    if 3 in opt.keys(): 
        x = clip_xval(x, opt[3])
    
    if opt[4] == 'upper':
        ymask=(y>=65)    
        x, y, idx = mask_yval(x, y, idx,  ymask)
    if opt[4] == 'under':
        ymask=(y<65)    
        x, y, idx = mask_yval(x, y, idx, ymask)
    
    
    l = get_label(y, opt[4])
    
    if 5 in opt.keys(): 
        x, y, l = balance_data(x, y, l, opt[5], tag_cut[opt[4]], rs=opt[8])
    
    if 6 in opt.keys():
        y = norm_y(y, opt[6])
    
    x, y, l, xv, yv, lv = split_data(x, y, l, opt[7], rs=opt[8])
    
    return x, y, l, xv, yv, lv

def main():
    
    data_path = sys.argv[1]
    save_path = '%s/%s'%(data_path, sys.argv[2])
    if not os.path.isdir(save_path): os.mkdir(save_path)
    sys.stdout = open('%s/alog.txt'%save_path,'w') 
    print('1:  rmidx, 2: rmy,  3: clipx, 4: tag, 5: balance, 6: normy, 7: split_size, 8: rs')
    
    opt = { 
          1: 24,       # option, remove idx
          2: 10,       # option, remove abnormal y (under 10)
          3: 100,      # option, clip x range
          4: 'upper', # * tag (class4, class2, upper, under)
          #5: 0.5,      # option, balance size(0.2, 0.3 ...)
          6: [65, 100], # option, normalize y (minmax, log)
          7: 0.2,      # * split data size
          8: 34,       # * random state
          }
    print(opt) 
    
    x, y, idx = load_data(data_path)
    print(x.shape, y.shape, idx.shape)
    
    x, y, l, xv, yv, lv = process(x, y, idx, opt)
    print(x.shape, y.shape, l.shape)
    print(xv.shape, yv.shape, lv.shape)

    save_data(x, y, l, save_path, tag='train') 
    save_data(xv, yv, lv, save_path, tag='valid')
    
    draw_lhist('l', [l, lv], tlist=['train set', 'valid set'], save=save_path+'/hist_l')
    draw_lhist('y', [y, yv], tlist=['train set', 'valid set'], save=save_path+'/hist_y')

if __name__ == '__main__':
    main()
