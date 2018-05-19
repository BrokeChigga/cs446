"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1, 1).
        """
        reg_grad = None
        loss_grad = None
        # Implementation here.
        pass
        (a, b) = self.x.shape
        reg_grad = np.zeros((b, 1))
        for i in range(a):
            #print(self.x.shape)
            #print(self.w.shape)
            if (y[i]*np.matmul(self.x[i], self.w)) < 1:
                #print(y[i].shape)
                #print(((self.x)[i]).shape)
                x_i = np.reshape((self.x)[i], (len((self.x)[i]),1))
                y_i = np.reshape(y[i], (1,1))
                #print(np.matmul(x_i, y_i).shape)
                reg_grad += -1*np.matmul(x_i, y_i)
        loss_grad = self.w_decay_factor*self.w

        total_grad = reg_grad + loss_grad
        return total_grad

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """

        hinge_loss = 0
        l2_loss = 0
        # Implementation here.
        pass
        for i in range(len(f)):
            y_i = y[i][0]
            f_i = f[i][0]
            #print(y_i*f_i)
            #print(1-y_i*f_i)
            hinge_loss += max(0, 1-y_i*f_i)
        #hinge_loss = max(0, 1-np.matmul(f.T, y))                
        l2_loss = 0.5*self.w_decay_factor*np.power(np.linalg.norm(self.w), 2)

        total_loss = hinge_loss + l2_loss
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        y_predict = None
        # Implementation here.
        pass
        y_predict = []
        N = f.shape[0]        
        for out in f:
            if np.sign(out) >=0:
                y_predict.append(1.0)
            else:
                y_predict.append(-1.0)
        y_predict = np.reshape(np.array(y_predict), (N,1))
        return y_predict
