import numpy as np

from modelos.gauss_tradicional import GaussTradicional

class GaussNaive(GaussTradicional):
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)

    # Quase igual ao do Gauss Tradicional. No lugar de calcular a matriz de covariância, calcula apenas a variância.
    # A fórmula é quese igual a do professor -> self.Sigma[i] = np.cov(self.X[i])
    def fit(self):
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.Sigma[i] = np.var(self.X[i], axis=1).reshape(self.p, 1) + 1e-6
            self.P[i] = self.n[i] / self.N

    def predict(self, X_teste_sample):
        scores = np.zeros(self.C)
        for i in range(self.C):
            termo_classe = np.log(self.P[i])
            var = self.Sigma[i] + 1e-12
            termo = -0.5 * np.sum(np.log(2 * np.pi * var)) \
                    -0.5 * np.sum(((X_teste_sample - self.mu[i])**2) / var)
            scores[i] = termo_classe + termo

        return np.argmax(scores) + 1
    
    def predict_batch(self, X_test):
        """
        Predição vetorizada para Naive Bayes Gaussiano.
        X_test: (p, N_te)  -> retorna (N_te,) em {1..C}
        Requer que no fit() você tenha:
        - self.mu[i] como (p,1)
        - self.Sigma[i] como variâncias (p,1), **não** matriz cheia
        - self.P[i] como prior escalar
        """
        p, N_te = X_test.shape
        C = self.C

        means = np.hstack(self.mu).T                 # (C, p)
        vars_ = np.hstack(self.Sigma).T              # (C, p)  # cada Sigma[i] é (p,1)
        vars_ = np.maximum(vars_, 1e-12)
        invvars = 1.0 / vars_
        logdet = np.sum(np.log(vars_), axis=1)       # (C,)
        logpri  = np.log(np.array(self.P) + 1e-12)   # (C,)

        X = X_test.T                                 # (N_te, p)
        diff = X[:, None, :] - means[None, :, :]     # (N_te, C, p)
        maha = np.einsum('ncp,cp->nc', diff**2, invvars)  # (N_te, C)

        scores = logpri[None,:] - 0.5*(p*np.log(2.0*np.pi) + logdet[None,:] + maha)
        return np.argmax(scores, axis=1) + 1
