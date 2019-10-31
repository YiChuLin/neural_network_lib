import numpy as np

'''
    The general function signature of initializer, which initialize the parameters(weights)

    Args:
        param_shape (tuple of int): the shape of parameters to initialize
        *args: Variable length argument list.

    Returns:
        initialized_param (np.ndarray): intialized ndarray with shape of param_shape
'''

def const_initializer(param_shape,c=0):
    return c*np.ones(param_shape)

def xavier_initializer(param_shape,input_num,output_num):
    low_limit=-np.sqrt(6/(input_num+output_num))
    high_limit=np.sqrt(6/(input_num+output_num))
    return np.random.uniform(low_limit,high_limit,param_shape)