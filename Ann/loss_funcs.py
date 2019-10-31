import numpy as np

'''
    The general function signature of loss functions, which return the gradient of the prediction

    Args:
        prediction (np.ndarray): the predictions made by the model, with shape batch_size x label_size
        ground_truth (np.ndarray): the label of each example in the batch, with shape batch_size x label_size
        *args: Variable length argument list.

    Returns:
        loss (float): loss value
        grad (np.ndarray): gradient of the prediction, grad.shape==prediction.shape
'''

def MSE_loss(prediction,ground_truth):
    assert(prediction.shape==ground_truth.shape)
    
    res=prediction-ground_truth
    loss=(res**2).sum()/res.size
    grad=2*res/res.size

    return loss,grad


'''
    The general function signature of regularisers, which the penalty on parameters

    Args:
        param (np.ndarray): the parameter in the model
        *args: Variable length argument list.

    Returns:
        grad (np.ndarray): gradient of the parameter from regularisation
'''

def L2_regularisation(param,lambda_value):
    return 2*lambda_value*param

def zero_regularisation(param):
    return np.zeros(param.shape)
    