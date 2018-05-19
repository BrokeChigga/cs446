"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression
from random import *

def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    pass
    #???
    
    copy_dataset = np.concatenate(processed_dataset, axis=1)
    #print(copy_dataset.shape)
    for i in range(num_steps):
        if shuffle:
            np.random.shuffle(copy_dataset)
            #print(copy_dataset[0])
        number_example = len(copy_dataset)
        number_batch = number_example // batch_size
        if (number_example%batch_size != 0):
            number_batch +=1
        batches = np.array_split(copy_dataset, number_batch)
        idx = randint(0,number_batch-1)
        select_batch = batches[idx]
        #print(select_batch.shape)
        batch_x = select_batch[:,0:-1]
        batch_y = select_batch[:,-1]
        batch_y = batch_y.reshape(batch_y.shape[0], 1)
        #print(batch_x.shape)
        #print(batch_y.shape)
        update_step(batch_x, batch_y, model, learning_rate)

    return model
    """
    #print(processed_dataset[0].shape)
    #print(processed_dataset[1].shape)
    doc_len = processed_dataset[0].shape[0]
    new_x = None
    new_y = None
    for steps in range(num_steps):
        # print("steps: ", steps)
        if shuffle:
            data_copy = np.concatenate(processed_dataset, axis=1)
           # print(data_copy.shape)
            np.random.shuffle(data_copy)
            new_x = data_copy[:, 0:-1]
            new_y = data_copy[:, -1]
            new_y = new_y.reshape(new_y.shape[0], 1)
        else:
            new_x = processed_dataset[0]
            new_y = processed_dataset[1]
        i = randint(0, int(doc_len/batch_size))
        if i*batch_size < doc_len:
            update_step(new_x[i*batch_size: (i+1)*batch_size], \
                new_y[i*batch_size: (i+1)*batch_size], \
                model, learning_rate)
        elif i*batch_size > doc_len:
            update_step(new_x[i*batch_size: doc_len], \
                new_y[i*batch_size: doc_len], \
                model, learning_rate)

    return model
    """

def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    pass
    for_res = model.forward(x_batch)
    back_res = model.backward(for_res, y_batch)
    model.w -= learning_rate*back_res


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    pass
    lamda = 0.1
    x = processed_dataset[0]
    #add one???
    (N,ndims) = x.shape
    one = np.array([np.array([1]*N)])
    x = np.concatenate((x,one.T),axis=1)    

    y = processed_dataset[1]
    temp_1 = np.matmul(x.T, x)
    (n,m) = temp_1.shape
    temp_2 = temp_1 + lamda*np.identity(n)
    temp_2 = np.linalg.inv(temp_2)
    model.w = np.matmul(np.matmul(temp_2, x.T), y)
    model.w = (model.w).reshape((len(model.w),1))
    #print("analytic w shape:")
    #print((model.w).shape)


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    loss = None
    #print(processed_dataset[0].shape)
    f = model.forward(processed_dataset[0])
    y = processed_dataset[1]
    loss = model.total_loss(f,y)
    return loss
