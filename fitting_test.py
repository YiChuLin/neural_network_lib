import numpy as np
import matplotlib.pyplot as plt

from Ann.networks import FittingNet
from Ann.loss_funcs import MSE_loss

if __name__=="__main__":

    k=1
    N=100
    
    a=0.7
    b=0.3
    c=1

    hidden_num_multiply=5
    train_iter=50000

    X=np.random.uniform(-10,10,(N*k))
    X=np.sort(X.flatten())
    X=X.reshape((N,1,k))

    Y=a*np.sin(X)+b*np.cos(X+c)

    Y=Y.reshape((N,1,1))

    plt.plot(X.flatten(),Y.flatten())
    plt.show()

    net=FittingNet(k,hidden_num_multiply)

    for i in range(0,train_iter):
        prediction_batch=net.forward(X)
        loss,loss_batch_grad=MSE_loss(prediction_batch,Y)
        print("iter "+str(i)+"-th loss="+str(loss))
        net.backward(loss_batch_grad)


    plt.scatter(X.flatten(),Y.flatten())

    X_test=np.linspace(-10,10,2000)
    X_test=X_test.reshape((2000,1,k))
    Y_test=net.forward(X_test)

    plt.plot(X_test.flatten(),Y_test.flatten())

    plt.show()
