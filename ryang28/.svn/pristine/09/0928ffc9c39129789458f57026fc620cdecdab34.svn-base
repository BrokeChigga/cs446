"""logistic model class for binary classification."""

import numpy as np

class LogisticModel(object):
    
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term, 
            self.W = [Bias, W1, W2, W3, ...] 
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        self.W = np.empty([ndims+1,1])
        if W_init == 'zeros':
            pass
            (self.W).fill(0)
        elif W_init == 'ones':
            pass
            (self.W).fill(1)
        elif W_init == 'uniform':
            pass
            (self.W) = np.random.uniform(0,1,ndims+1)
            self.W = np.reshape(self.W, (ndims+1,1))
            #(self.W).fill(number)
        elif W_init == 'gaussian':
            pass
            self.W = np.random.normal(0, 0.1, ndims+1)
            self.W = np.reshape(self.W, (ndims+1,1))
        else:
            print ('Unknown W_init ', W_init) 
        
    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print ('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print ('model loaded from', weight_file)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for each sample 
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        pass
        score = np.matmul(X, self.W)
        a = X.shape[0]
        score = np.reshape(score, (a,))
        score = 1/(1+np.exp(-1*score))
        return score

    def backward(self, Y_true, X):
        """ Backward operation for logistic models. 
            Compute gradient according to the probability loss on lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        pass
        """
        a,b = Y_true.shape
        Y = np.reshape(Y_true, (a,1))
        exp = np.exp(-1*np.matmul(np.matmul(Y.T, X), self.W))
        gradient = -1*np.matmul(X.T, Y)*exp / (1+exp)
        return gradient
        """
        gradient = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            #print(type(X[i][0]))
            #print(type((self.W)[0]))
            exp = np.dot(X[i], self.W)
            exp = np.exp(-1*Y_true[i]*exp)
            exp = exp/(1+exp)
            temp = -1*Y_true[i]*exp
            temp = temp * X[i]
            gradient = gradient + temp
        return gradient

    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        pass
        score = self.forward(X)
        predict_label = []
        for s in score:
            if s>= 0.5:
                predict_label.append(1)    
            else:
                predict_label.append(-1)
        return np.array(predict_label)        

    def fit(self, Y_true, X, learn_rate, max_iters):
        """ train model with input dataset using gradient descent. 
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        pass
        for i in range(max_iters):
            g = self.backward(Y_true, X)
            g = np.reshape(g, (self.W.shape[0],1))
            #print(g.shape)
            #print(self.W.shape)
            self.W -= learn_rate*g

