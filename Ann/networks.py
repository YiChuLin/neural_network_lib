from network_proto import Network
import layers
import initializers
import learners
import loss_funcs

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
        return learners.gradient_descent(param,param_grad,0.2)

    def regulariser(self,param):
        return loss_funcs.zero_regularisation(param)

