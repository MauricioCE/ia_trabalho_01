import numpy as np

class GaussianClassifier:
    def __init__(self,X_train, y_train):
        self.classes = np.unique(y_train)
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        self.X = [X_train[:,y_train[0,:]==i] for i in self.classes]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        self.Sigma = [None]*self.C
        self.Sigma_det = [None]*self.C
        self.Sigma_inv = [None]*self.C
        self.mu = [None]*self.C
        self.P = [None]*self.C
        
        self.g = [None]*self.C
        
    def fit(self):
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i],axis=1).reshape(self.p,1)
            self.Sigma[i] = np.cov(self.X[i])
            self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
            self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])
            self.P[i] = self.n[i]/self.N
    
    def predict(self, x_test):
        
        for i in range(self.C):
            d_mahalanobis = ((x_test-self.mu[i]).T@self.Sigma_inv[i]@(x_test-self.mu[i]))[0,0]
            self.g[i] = np.log(self.P[i]) - 1/2*np.log(self.Sigma_det[i]) - 1/2*d_mahalanobis
        return np.argmax(self.g)+1            