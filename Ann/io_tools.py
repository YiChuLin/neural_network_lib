import pickle
import numpy as np

from Ann.network_proto import Network

def serialize_params(net):
    assert(Network in type(net).__bases__)
    net_params=[]
    for layer in net.layer_list:
        layer_params=[]
        for layer_param in layer.params:
            layer_params.append(layer_param)
        net_params.append(layer_params)
    
    return net_params


def save_params(net,file_name):
    with open(file_name,'wb') as bin_file:
        pickle.dump(serialize_params(net),bin_file)


'''
    if network load parameters from a binary file before forward/backward, initializers will not be called
'''
def load_params(net,file_name):
    with open(file_name,'rb') as bin_file:
        loaded_layers=pickle.load(bin_file)
        # check compatibility and load
        assert len(loaded_layers)==len(net.layer_list)
        for loaded_params,layer in zip(loaded_layers,net.layer_list):
            assert len(loaded_params)==len(layer.param_shapes)
            assert not layer.is_initialized and len(layer.params)==0
            for param,param_shape in zip(loaded_params,layer.param_shapes):
                assert type(param)==np.ndarray
                assert param.shape==param_shape
                layer.params.append(param)
            layer.is_initialized=True

