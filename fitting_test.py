import numpy as np
import matplotlib.pyplot as plt

from Ann.networks import FittingNet
from Ann.loss_funcs import MSE_loss

if __name__=="__main__":

    k=1
    N=20

    hidden_num_multiply=5
    train_iter=10000

    X=np.random.uniform(-1,1,(N*k))
    X=np.sort(X.flatten())
    X=X.reshape((N,1,k))

    Y=np.sin(X)

    Y=Y.reshape((N,1,1))

    # find pre-defined networks in Ann.networks
    net=FittingNet(k,hidden_num_multiply)

    for i in range(0,train_iter):
        prediction_batch=net.forward(X)
        # other loss functions in Ann.loss_funcs
        loss,loss_batch_grad=MSE_loss(prediction_batch,Y)
        print("iter "+str(i)+"-th loss="+str(loss))
        net.backward(loss_batch_grad)


    plt.scatter(X.flatten(),Y.flatten())

    X_test=np.linspace(-1,1,100)
    X_test=X_test.reshape((100,1,k))
    Y_test=net.forward(X_test)

    plt.plot(X_test.flatten(),Y_test.flatten())

    plt.show()
