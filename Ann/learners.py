import numpy as np

'''
    The general function signature of learner, which update the parameters by gradients

    Args:
        param (np.ndarray): the current value of the parameters
        param_grad (np.ndarray): the gradients of the parameters, param_grad.shape==param.shape
        *args: Variable length argument list.

    Returns:
        updated_params (np.ndarray): updated parameters, updated_params.shape==param.shape
'''


def gradient_descent(param,param_grad,learning_rate):
    return param-learning_rate*param_grad

def gradient_descent_with_momentum(param,param_grad,beta):
    assert(beta<1 and beta>0)
    return beta*param+(1-beta)*param_grad