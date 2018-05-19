"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = None
max_iters = None

def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')
    A, T = read_dataset_tf('../data/trainset','indexing.txt')
    # Initialize model.
    model = LogisticModel_TF(ndims = 16, W_init = 'gaussian')
    # Build TensorFlow training graph
    model.build_graph(0.001)
    # Train model via gradient descent.
    ret_label = model.fit(T, A, 1000)
    # Compute classification accuracy based on the return of the "fit" method
    #print(ret_label)
    train_label = []
    for i in range(ret_label.shape[0]):    
        if ret_label[i][0] >= 0.5:
            train_label.append(1)
        else:
            train_label.append(-1)
    correct = 0
    for i in range(len(T)):
        if T[i] == train_label[i]:
            correct += 1
    accuracy = correct/len(T)
    print(accuracy)
    pass 

    
if __name__ == '__main__':
    tf.app.run()
