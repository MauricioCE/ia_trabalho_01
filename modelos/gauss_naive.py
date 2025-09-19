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
            self.Sigma[i] = np.var(self.X[i], axis=1).reshape(self.p, 1) # Só isso
            self.P[i] = self.n[i] / self.N

    def predict(self, X_teste_sample):
        scores = np.zeros(self.C)
        for i in range(self.C):
            termo_classe = np.log(self.P[i])
            termo = -0.5 * np.sum(np.log(2 * np.pi * self.Sigma[i])) \
                                    -0.5 * np.sum(((X_teste_sample - self.mu[i])**2) / self.Sigma[i])
            scores[i] = termo_classe + termo

        return np.argmax(scores) + 1