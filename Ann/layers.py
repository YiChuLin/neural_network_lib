import numpy as np

from layer_proto import Layer


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

