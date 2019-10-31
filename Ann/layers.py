import numpy as np

from Ann.layer_proto import Layer

'''
    The general structure of a derived class of Ann.layer_proto.Layer

    The following attributes can be used for computation.
    The four methods must be implemented.

    Attributes
    ----------
        input_shape: tuple, inherited
            the shape of the input(single example, not batch)

        output_shape: tuple, inherited
            the shape of the output(single example, not batch)

        param_shapes: list of tuples, inherited
            the shape of each parameter ndarray

        params: list of ndarrays, inherited
            the parameter ndarray

    Methods
    -------
    __init__(arg*)
        must initialize the parent class(Layer) using (input_shape,param_shapes,output_shape)

    compute_forward(input):
        input: ndarray, same shape as input_shape 
        return the output, a ndarray with same shape as output_shape

    compute_params_grad(output_grad,input,output):
        output_grad: ndarray, same shape as output_shape
            the gradient of the output
        input: ndarray, same shape as input_shape
            the input in forwarding
        output: ndarray, same shape as output_shape
            the output in forwarding        
        return the gradient of the parameters
            a list of ndarrays with same shape w.r.t. each param in params.

    compute_input_grad(output_grad,input):
        output_grad: ndarray, same shape as output_shape
            the gradient of the output
        input: ndarray, same shape as input_shape
            the input in forwarding        
        return the gradient of the input
            an ndarrays with same shape as input_shape.

'''

class Relu(Layer):
    def __init__(self,size):
        super().__init__((1,size),[],(1,size))

    def compute_forward(self,input):
        X=input
        X[X<0]=0
        return X

    def compute_params_grad(self,output_grad,input,output):
        pass

    def compute_input_grad(self,output_grad,input):
        X=input
        X[X>=0]=1
        X[X<0]=0
        return input*output_grad


class Sigmoid(Layer):
    def __init__(self,size):
        super().__init__((1,size),[],(1,size))

    def compute_forward(self,input):
        return np.exp(input)/(np.exp(input)+1)

    def compute_params_grad(self,output_grad,input,output):
        pass

    def compute_input_grad(self,output_grad,input):
        return output_grad*np.exp(input)/((np.exp(input)+1)**2)


class Linear(Layer):
    def __init__(self,input_size,output_size):
        assert type(input_size)==int and input_size>0
        assert type(output_size)==int and output_size>0

        input_shape=(1,input_size)
        output_shape=(1,output_size)
        param_shapes=[(input_size,output_size),(1,output_size)]
        
        super().__init__(input_shape,param_shapes,output_shape)

    def compute_forward(self,input):
        x=input
        W=self.params[0]
        d=self.params[1]
        return x.dot(W)+d

    def compute_params_grad(self,output_grad,input,output):
        return [input.T.dot(output_grad),output_grad]

    def compute_input_grad(self,output_grad,input):
        return output_grad.dot(self.params[0].T)

