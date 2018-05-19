import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        pass
        ret = {}
        labels = np.unique(y)
        for l in labels:
            y_train = y.copy()
            y_train = [1 if item == l else 0 for item in y_train]
            #shape???
            y_train = np.array(y_train)
            clf = svm.LinearSVC(random_state=12345)
            clf.fit(X, y_train)
            ret[l] = clf
        return ret        

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        pass
        ret = {}
        labels = np.unique(y)
        pairs = []
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                pairs.append([i,j])
        for p in pairs:
            label_1 = p[0]
            label_2 = p[1]
            X_train = []
            y_train = []
            for i in range(len(y)):
                if y[i] == label_1:
                    X_train.append(X[i])
                    #first label as 0
                    y_train.append(0)
                elif y[i] == label_2:
                    X_train.append(X[i])
                    #second label as 1
                    y_train.append(1)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            clf = svm.LinearSVC(random_state=12345)
            clf.fit(X_train, y_train)    
            ret[(label_1, label_2)] = clf
        return ret

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        pass
        a = X.shape[0]
        b = len(self.labels)
        scores = np.zeros((a, b))

        for i in range(len(self.labels)):
            curr_label = self.labels[i]
            clf = self.binary_svm[curr_label]
            temp_score = clf.decision_function(X)
            #shape???
            scores[:,i] = temp_score
        scores = np.array(scores)
        return scores


    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        pass
        a = X.shape[0]
        b = len(self.labels)
        scores = np.zeros((a, b))

        for label_pair in self.binary_svm:
            label_1 = label_pair[0] #0
            label_2 = label_pair[1] #1
            clf = self.binary_svm[label_pair]
            predict_label = clf.predict(X)
            for i in range(len(predict_label)):
                if predict_label[i] == 0:
                    scores[i][label_1] += 1
                elif predict_label[i] == 1:
                    scores[i][label_2] += 1
        return scores

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        pass
        K = W.shape[0]
        N = X.shape[0]
        reg = 0
        loss = 0
        total_loss = 0

        for i in range(K):
            reg += 0.5*np.sum(np.square(W[i]))

        for i in range(N):
            '''
            curr_max = float('-inf')
            for j in range(K):
                curr_sum = 1
                if j == y[i]:
                    curr_sum -= 1
                curr_sum += np.matmul(X[i], W[j])
                if curr_sum > curr_max:
                    curr_max = curr_sum
            '''
            row_k1 = np.dot(W, X[i])
            for j in range(K):
                if j != y[i]:
                    row_k1[j] += 1
            curr_max = np.max(row_k1)
            loss += C * (curr_max - np.matmul(X[i], W[y[i]]))
        #print(reg)
        #print(loss)
        total_loss = reg + loss
        return total_loss

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        pass
        K = W.shape[0]
        N = X.shape[0]
        #print(K)
        #print(N)
        gradient = W.copy()
        for i in range(N):
            '''
            curr_max = float('-inf')
            max_j = -1
            for j in range(K):
                curr_sum = 1
                if j == y[i]:
                    curr_sum -= 1
                curr_sum += np.matmul(X[i], W[j])
                if curr_sum > curr_max:
                    curr_max = curr_sum
                    max_j = j
                    #max_i = i
            '''
            row_k1 = np.dot(W, X[i])
            for j in range(K):
                if j != y[i]:
                    row_k1[j] += 1
            max_j = np.argmax(row_k1)
            gradient[max_j] += C * X[i]
            gradient[y[i]] -= C * X[i]

        return gradient
