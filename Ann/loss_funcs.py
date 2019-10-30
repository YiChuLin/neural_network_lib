import numpy as np

def MSE_loss(prediction,ground_truth):
    assert(prediction.shape==ground_truth.shape)
    
    res=prediction-ground_truth
    loss=(res**2).sum()/res.size
    grad=2*res/res.size

    return loss,grad
    

def L2_regularisation(param,lambda_value):
    return 2*lambda_value*param

def zero_regularisation(param):
    return np.zeros(param.shape)
    