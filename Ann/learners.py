import numpy as np

def gradient_descent(param,param_grad,learning_rate):
    return param-learning_rate*param_grad

def gradient_descent_with_momentum(param,param_grad,beta):
    assert(beta<1 and beta>0)
    return beta*param+(1-beta)*param_grad