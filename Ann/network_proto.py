from layer_proto import Layer

class Network:
    def __init__(self,layer_list):
        assert(type(layer_list)==list and len(layer_list)!=0)
        for i,layer in enumerate(layer_list):
            assert(Layer in type(layer).__bases__)
            if i>0:
                assert(last_layer_output==layer.input_shape)
            last_layer_output=layer.output_shape
        

        self.layer_list=layer_list

        self.set_layer_initializer()

    def forward(self,input_batch):
        for layer in self.layer_list:
            output_batch=layer.forward(input_batch)
            input_batch=output_batch
        
        return output_batch

    def learner(self,param,param_grad):
        raise NotImplementedError

    def regulariser(self,param):
        raise NotImplementedError

    def set_layer_initializer(self):
        raise NotImplementedError

    def backward(self,grad):
        for layer in self.layer_list[::-1]:
            grad=layer.backward(grad,self.learner,self.regulariser)
        
