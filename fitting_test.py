import numpy as np
import matplotlib.pyplot as plt

from Ann.networks import FittingNet
from Ann.loss_funcs import MSE_loss

# This test trains a network to fit a function

if __name__=="__main__":

    # number of input
    k=1
    # number of training examples
    N=20
    # hidden_layer_size/input_size
    hidden_layer_multiply=5
    # training iterations
    train_iter=2000

    # generate training examples
    X=np.random.uniform(-1,1,(N*k))
    X=X.reshape((N,1,k)) # input shaped to batch_size x input_shape
    Y=np.sin(X)
    Y=Y.reshape((N,1,1)) # output shaped to batch_size x output_shape

    # find pre-defined networks in Ann.networks
    net=FittingNet(k,hidden_layer_multiply)

    # training
    for i in range(0,train_iter):
        prediction_batch=net.forward(X)
        # find loss functions in Ann.loss_funcs
        loss,loss_batch_grad=MSE_loss(prediction_batch,Y)
        print("iter "+str(i)+"-th loss="+str(loss))
        net.backward(loss_batch_grad)

    # plot training examples
    plt.scatter(X.flatten(),Y.flatten())

    # generate test data
    X_test=np.linspace(-1,1,100)
    X_test=X_test.reshape((100,1,k))
    # make preditions using test data
    Y_test=net.forward(X_test)

    plt.plot(X_test.flatten(),Y_test.flatten())

    plt.show()
