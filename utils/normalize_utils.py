################################################
# Not used. Useful if visualization is desired #
################################################

import numpy as np 

mean_vec=[0.485, 0.456, 0.406]
std_vec=[0.229, 0.224, 0.225]

def normalize_np(data,mean,std):
    #input data B*W*H*C
    B,W,H,C=data.shape
    mean=np.array(mean).reshape([1,1,1,C])
    std=np.array(std).reshape([1,1,1,C])
    return (data-mean)/std

def unnormalize_np(data,mean,std):
    #input data B*W*H*C
    B,W,H,C=data.shape
    mean=np.array(mean).reshape([1,1,1,C])
    std=np.array(std).reshape([1,1,1,C])
    return data*std+mean