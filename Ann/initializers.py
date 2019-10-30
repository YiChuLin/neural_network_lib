import numpy as np

def const_initializer(param_shape,c=0):
    return c*np.ones(param_shape)

def xavier_initializer(param_shape,input_num,output_num):
    low_limit=-np.sqrt(6/(input_num+output_num))
    high_limit=np.sqrt(6/(input_num+output_num))
    return np.random.uniform(low_limit,high_limit,param_shape)