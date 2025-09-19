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

    def predict(self, X_teste_sample):
        Sigma_inv = np.linalg.inv(self.Sigma)
        log_det_Sigma = np.log(np.linalg.det(self.Sigma))

        pontuacoes = np.zeros(self.C)
        for i in range(self.C):
            prob_a_priori = np.log(self.P[i])
            diferenca_media = X_teste_sample.reshape(self.p, 1) - self.mu[i]
            termo = -0.5 * log_det_Sigma - 0.5 * np.dot(np.dot(diferenca_media.T, Sigma_inv), diferenca_media)
            pontuacoes[i] = prob_a_priori + termo

        return np.argmax(pontuacoes) + 1