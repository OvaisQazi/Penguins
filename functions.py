import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

# neural network dimensions
n0 = 4 # number of features = input layer
n1 = 20 # middle layer
n2 = 3 # number of labels = output layer

# changing vector of weights into matrices
def wvec_to_wmats(wvec, n0 = n0, n1 = n1, n2 = n2):
    windex = n0 * n1
    a1 = wvec[:windex].reshape((n1, n0)) # weights for input -> middle layer
    b1 = wvec[windex : windex+n1] # weights for bias for input -> middle layer
    windex += n1
    a2 = wvec[windex : windex + (n1*n2)].reshape((n2, n1)) # weights for middle -> output layer
    windex += n1 * n2
    b2 = wvec[windex:] # weights for bias for middle -> output layer

    if (a1.size + b1.size + a2.size + b2.size) != wvec.size:
        raise UserWarning('mismatch weightsvector/matrices')
    
    return a1, b1, a2, b2

# neural network
def fwdnn(x0, a1=None, b1=None, a2=None, b2=None):
    x1 = np.tanh(a1 @ x0 + b1) # calculate middle layer
    x2 = np.tanh(a2 @ x1 + b2) # calculate output layer
    return x2

# loss function
def sqrdloss(wvec, x=None, y=None):
    a1, b1, a2, b2 = wvec_to_wmats(wvec)
    y_pred = fwdnn(x, a1, b1, a2, b2)
    return np.sum((y - y_pred) ** 2)

# training loop
def nn(lr, epochs, ss_sgd, wvec, x_train, y_train, batch_size):
    lh = []
    gnh = []
    n = len(x_train)

    for i in range(epochs):
        p = np.random.permutation(n)
        x_train = x_train[p]
        y_train = y_train[p]

        for j in range(0, n, batch_size):
            x_batch = x_train[j : j+batch_size]
            y_batch = y_train[j : j+batch_size]
            
            grad = np.mean(np.array([approx_fprime(wvec, sqrdloss, ss_sgd, x_batch[k], y_batch[k])
                for k in range(len(x_batch))]), axis = 0)

            wvec -= lr * grad
        
        total_loss = np.mean([sqrdloss(wvec, x_train[j], y_train[j]) for j in range(len(x_train))])
        grad_norm = np.linalg.norm(grad)
        lh.append(total_loss)
        gnh.append(grad_norm)

        print(f"epoch {i:5d} | loss = {total_loss:.4f} | ||grad|| = {grad_norm:.4f}")    

    print("Training done")

    return lh, gnh, wvec

# predicting on testing data
def predict(wvec, x_test):
    a1, b1, a2, b2 = wvec_to_wmats(wvec)
    pred = []
    for x in x_test:
        y_pred = fwdnn(x, a1, b1, a2, b2)
        pred.append(np.argmax(y_pred))
    return np.array(pred)