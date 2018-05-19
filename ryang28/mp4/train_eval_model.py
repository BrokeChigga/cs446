"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers
from random import *

def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    pass
    #print(data['image'].shape)
    #print(data['label'].shape)
    copy_dataset = np.concatenate((data['image'], data['label']), axis=1)
    for i in range(num_steps):
        if shuffle:
            np.random.shuffle(copy_dataset)
        number_example = len(copy_dataset)
        number_batch = number_example // batch_size
        if (number_example%batch_size != 0):
            number_batch +=1
        batches = np.array_split(copy_dataset, number_batch)
        idx = randint(0,number_batch-1)
        select_batch = batches[idx]
        batch_x = select_batch[:,0:-1]
        batch_y = select_batch[:,-1]
        batch_y = batch_y.reshape(batch_y.shape[0], 1)
        update_step(batch_x, batch_y, model, learning_rate)    

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    pass
    for_res = model.forward(x_batch)
    back_res = model.backward(for_res, y_batch)
    model.w -= learning_rate*back_res

def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    pass
    # Set model.w
    (N, n_1) = model.x.shape
    model.w = z[:n_1]

def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    P = None
    q = None
    G = None
    h = None
    # Implementation here.
    x = data['image']
    (N,ndims) = x.shape
    one = np.array([np.array([1]*N)])
    model.x = np.concatenate((x,one.T),axis=1)
    (N, n_1) = model.x.shape
    x = model.x
    y = data['label']
    #print(N, n_1)
    P = np.zeros((N+n_1, N+n_1))
    for i in range (n_1):
        P[i][i] = model.w_decay_factor

    q = np.zeros((N+n_1, ))
    q[n_1:] = 1
    q = np.reshape(q, (N+n_1, 1))

    G = np.zeros((2*N, N+n_1))
    for i in range (N):
        for j in range (N+n_1):
            if j < n_1:
                G[i][j] = -1*y[i]*x[i][j]
            elif j-n_1 == i:
                G[i][j] = -1
    for i in range(N, 2*N):
        G[i][n_1+i-N] = -1
    print(G.shape)

    h = np.zeros((2*N, ))
    h[:N] = -1
    h = np.reshape(h, (2*N, 1))
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    loss = 0
    acc = 0
    f = model.forward(data['image'])
    y = data['label']
    loss = model.total_loss(f,y)

    ret_label = model.predict(f)
    correct = 0
    for i in range(len(ret_label)):
        if ret_label[i] == y[i]:
            correct += 1
    acc = correct/len(ret_label)
    return loss, acc
