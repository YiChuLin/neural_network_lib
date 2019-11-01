from Ann.network_proto import Network
import Ann.layers as layers
import Ann.initializers as initializers
import Ann.learners as learners
import Ann.loss_funcs as loss_funcs

'''
    The general structure of a derived class of Ann.network_proto.Network

    The four methods to be implemented.

    Methods
    -------
    __init__(arg*)
        must initialize the parent class(Network) using (layer_list)

    set_layer_initializer():
        set an initialization method for each layer in the layer_list
        find the intializers in Ann.intializers
        set layer.initializer to a callable
        
    learner(param,param_grad):
        the method to update network parameters(weights)
        find a learner function in Ann.learners
        return the learner result
    
    regulariser(param):
        the method to compute the gradient parameters from regularisation loss
        find a regulariser in Ann.loss_funcs
        return the regulariser result
    
'''


class SimpleNet(Network):
    def __init__(self):        
        self.layer_list=[
            layers.Linear(4,10),
            layers.Relu(10),
            layers.Linear(10,1),
            layers.Relu(1)]

        super().__init__(self.layer_list)

    def set_layer_initializer(self):
        for layer in self.layer_list:
            if type(layer)==layers.Linear:
                layer.initializer=lambda param_shape: initializers.const_initializer(param_shape,1)
        

    def learner(self,param,param_grad):
        return learners.gradient_descent(param,param_grad,0.1)

    def regulariser(self,param):
        return loss_funcs.L2_regularisation(param,2)


class FittingNet(Network):
    def __init__(self,N,k):    
        self.N=N
        self.k=k

        self.layer_list=[
            layers.Linear(N,k*N),
            layers.Sigmoid(k*N),
            layers.Linear(k*N,k*N),
            layers.Sigmoid(k*N),
            layers.Linear(k*N,1)]

        super().__init__(self.layer_list)

    def set_layer_initializer(self):
        for layer in self.layer_list:
            if type(layer)==layers.Linear:
                input_num=layer.input_shape[1]
                output_num=layer.output_shape[1]
                layer.initializer=lambda param_shape: initializers.xavier_initializer(param_shape,input_num,output_num)

    def learner(self,param,param_grad):
        return learners.gradient_descent(param,param_grad,0.1)

    def regulariser(self,param):
        return loss_funcs.zero_regularisation(param)

