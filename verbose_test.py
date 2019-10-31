import numpy as np

from Ann.networks import SimpleNet

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

    input_batch=np.array([[[1,2]],[[3,4]],[[5,6]]])
    print("Forward with input:")
    print(input_batch)
    prediction_batch=net.forward(input_batch)

    print("Predictions:")
    print(prediction_batch)

    label_batch=8*np.ones((3,1,1))
    print("Labels:")
    print(label_batch)
    
    from loss_funcs import MSE_loss
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
    
    
