import numpy as np

from Ann.networks import SimpleNet
from Ann.loss_funcs import MSE_loss

# This test print all the parameters, inputs and outputs in one forward-backward pass

if __name__=="__main__":
    # find pre-defined networks in Ann.networks
    net=SimpleNet()
    
    print("The layers of the network before initialization")
    for i,layer in enumerate(net.layer_list):
        print("The "+str(i)+"-th layer, type="+str(type(layer)))
        print("  param shapes:")
        for param_shape in layer.param_shapes:
            print("    "+str(param_shape))
        layer._initialize_params()
    
    print("The layers of the network after initialization")
    for i,layer in enumerate(net.layer_list):
        print("The "+str(i)+"-th layer, type="+str(type(layer)))
        print("  params:")
        for param in layer.params:
            print(param)

    input_batch=np.random.rand(1,1,4)
    print("Forward with input:")
    print(input_batch)
    prediction_batch=net.forward(input_batch)

    print("Predictions:")
    print(prediction_batch)

    label_batch=np.random.rand(1,1,1)
    print("Labels:")
    print(label_batch)
    
    loss,loss_batch_grad=MSE_loss(prediction_batch,label_batch)
    
    print("Loss:")
    print(loss)
    print("Loss Grad:")
    print(loss_batch_grad)

    print("Backward with label")
    net.backward(loss_batch_grad)

    print("The layers of the network after backward")
    for i,layer in enumerate(net.layer_list):
        print("The "+str(i)+"-th layer, type="+str(type(layer)))
        print("  params:")
        for param in layer.params:
            print(param)
    
    
