import numpy as np
from modelos.gauss_tradicional import GaussTradicional

class GaussCovarianciaAgregada(GaussTradicional):
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)

    def fit(self):
        # Igual ao tradicional, por classe
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.P[i] = self.n[i] / self.N

        # A matriz de covariância agregada
        Sigma_agregada = np.zeros((self.p, self.p))
        for i in range(self.C):
            Sigma_agregada += (self.n[i] - 1) * np.cov(self.X[i])

        self.Sigma = Sigma_agregada / (self.N - self.C)
        self.Sigma = self.Sigma + 1e-6 * np.eye(self.p)

    def predict(self, X_teste_sample):
        Sigma_inv = np.linalg.pinv(self.Sigma)
        sign, log_det_Sigma = np.linalg.slogdet(self.Sigma)
        if sign <= 0:
            self.Sigma = self.Sigma + 1e-6 * np.eye(self.p)
            Sigma_inv = np.linalg.pinv(self.Sigma)
            sign, log_det_Sigma = np.linalg.slogdet(self.Sigma)

        pontuacoes = np.zeros(self.C)
        for i in range(self.C):
            prob_a_priori = np.log(self.P[i])
            diferenca_media = X_teste_sample.reshape(self.p, 1) - self.mu[i]
            termo = -0.5 * log_det_Sigma - 0.5 * np.dot(np.dot(diferenca_media.T, Sigma_inv), diferenca_media)
            pontuacoes[i] = prob_a_priori + termo

        return np.argmax(pontuacoes) + 1
    
    def predict_batch(self, X_test):
        """
        Predição vetorizada para Gauss com covariância POOLED (igual para classes).
        X_test: (p, N_te) -> (N_te,)
        Requer:
        - self.Sigma (p,p), e calcular inv/logdet aqui
        - self.mu[i] (p,1), self.P[i]
        """
        p, N_te = X_test.shape
        C = self.C

        means = np.hstack(self.mu).T                 # (C, p)

        # inv + slogdet da pooled:
        Sigma = self.Sigma + 1e-12*np.eye(p)
        invS  = np.linalg.pinv(Sigma)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            Sigma = Sigma + 1e-6*np.eye(p)
            invS  = np.linalg.pinv(Sigma)
            sign, logdet = np.linalg.slogdet(Sigma)

        logpri = np.log(np.array(self.P) + 1e-12)    # (C,)

        X = X_test.T
        diff = X[:, None, :] - means[None, :, :]     # (N_te, C, p)
        maha = np.einsum('ncp,pq,ncq->nc', diff, invS, diff)

        scores = logpri[None,:] - 0.5*(p*np.log(2.0*np.pi) + logdet + maha)
        return np.argmax(scores, axis=1) + 1
