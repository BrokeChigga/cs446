"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        #self._reg_covar = 0.1
        self._reg_covar = 1000
        # Randomly Initialize model parameters
        #self._mu = np.zeros((n_components, n_dims))  # np.array of size (n_components, n_dims)
        self._mu = np.random.randint(10,size = (self._n_components,self._n_dims),)
        # Initialized with uniform distribution.
        self._pi = np.random.uniform(0, 1, n_components)  # np.array of size (n_components, 1)
        self._pi = self._pi / np.sum(self._pi)
        self._pi = np.reshape(self._pi, (n_components, 1))

        # Initialized with identity.
        self._sigma = []  # np.array of size (n_components, n_dims, n_dims)
        k = 500
        for i in range(n_components):
            self._sigma.append(k*np.identity(n_dims))
        self._sigma = np.array(self._sigma)
        self._sigma = self._sigma * self._reg_covar

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        pass
        # initialize self._mu with random points from the dataset for MNIST(override)
        self._mu = x[np.random.choice(x.shape[0], self._n_components, replace=False), :]
        for i in range(self._max_iter):
            print("iter = ", i)
            z_ik = self._e_step(x)
            self._m_step(x, z_ik)

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        return self.get_posterior(x)

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        pass
        N = x.shape[0]
        # np.array of size (n_components, 1)
        self._pi = np.sum(z_ik, axis=0)/N
        self._pi = np.reshape(self._pi, (self._n_components,1))
        # np.array of size (n_components, n_dims)
        #for j in range(self._n_components):
        #    self._mu[j] = np.dot(z_ik[:,j].T, x)/(N*self._pi[j])
        for j in range(self._n_components):
            for a in range(self._n_dims):
                self._mu[j,a] = (np.dot(z_ik[:,j],x[:,a]))/(N*self._pi[j])

        # np.array of size (n_components, n_dims, n_dims)
        for j in range(self._n_components):
            nominator = []
            for i in range(N):
                x_mu = x[i]-self._mu[j] #(1,n_dims)
                nominator.append(z_ik[i][j]*np.dot(x_mu.T, x_mu)) #(n_dims, n_dims)
            self._sigma[j] = np.sum(nominator)/(N*self._pi[j])

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """
        ret = None
        ret = []
        #N = x.shape[0]
        #for i in range(N):
        #    row = []
        #    for j in range(self._n_components):
        #        y = self._pi[j] * multivariate_normal.pdf(x[i], self._mu[j], self._sigma[j])
        #        row.append(y)
        #    ret.append(row)
        #return np.array(ret)
        N = x.shape[0]
        ret = []
        for j in range(self._n_components):
            for a in range(len(self._sigma[j])):
                self._sigma[j,a,a] += self._reg_covar
            N_func = self._multivariate_gaussian(x, self._mu[j], self._sigma[j])
            ret.append(self._pi[j]*N_func)
        return np.array(ret).T

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        A = self.get_conditional(x)
        marginals = np.sum(A, axis =1)
        return marginals

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_ik = []
        N = x.shape[0]
        A = self.get_conditional(x)
        M = self.get_marginals(x)
        for i in range(N):
            row = A[i]/M[i]
            z_ik.append(row)
        return np.array(z_ik)

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """

        self.cluster_label_map = []
        pass
        self.cluster_label_map = [0]*self._n_components
        self.fit(x)
        #z_ik, dimension (N, n_components)
        z_ik = self._e_step(x)
        N = x.shape[0]
        clusters = {}
        for i in range(N):
            idx = np.argmax(z_ik[i])
            if idx in clusters:
                clusters[idx].append(y[i])
            else:
                clusters[idx] = [y[i]]

        for i in range(self._n_components):
            if i not in clusters:
                self.cluster_label_map[i] = np.random.choice(y)
            else:
                counts = np.bincount(clusters[i])
                label = np.argmax(counts)
                self.cluster_label_map[i] = label

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        z_ik = self.get_posterior(x)
        y_hat = []
        
        N = x.shape[0]
        for i in range(N):
            idx = np.argmax(z_ik[i])
            label = self.cluster_label_map[idx]
            y_hat.append(label)        

        return np.array(y_hat)
