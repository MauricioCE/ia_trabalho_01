from modelos.gauss_tradicional import GaussTradicional
import numpy as np

# Seguindo o pseudocódigo do slide
# Reaproveitando a classe do Gauss Tradicional
class GaussCovarianciasGlobal(GaussTradicional):
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)

        # Matriz de covariância agregada global
        self.matriz_covariancia_agregada = None
        self.matriz_covariancia_inv_agregada = None
        self.matriz_covariancia_det_agregada = None

    def fit(self):
        # Agora só calcula as médias e probablidades para cada classe
        # Usando a mesma forma que o Gauss Tradicional do prfessor
        for i in range(self.C):
            # Calcula a média
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            # Calcula a probabilidade a priori
            self.P[i] = self.n[i] / self.N
        
        # Agora, ao invés de ser para cada classe, é uma só matriz de covariância
        matriz_dispersao = np.zeros((self.p, self.p))

        for i in range(self.C):
            diferenca = self.X[i] - self.mu[i]
            matriz_dispersao += np.dot(diferenca, diferenca.T)

        self.matriz_covariancia_agregada = matriz_dispersao / (self.N - self.C)
        self.matriz_covariancia_inv_agregada = np.linalg.pinv(self.matriz_covariancia_agregada)
        self.matriz_covariancia_det_agregada = np.linalg.det(self.matriz_covariancia_agregada)

    def predict(self, x_test):
        # Calcula a função discriminante para cada classe
        for i in range(self.C):
            diferenca = x_test - self.mu[i]
            d_mahalanobis = (diferenca.T @ self.matriz_covariancia_inv_agregada @ diferenca)[0, 0]
            self.g[i] = (np.log(self.P[i]) - 0.5 * np.log(self.matriz_covariancia_det_agregada) - 0.5 * d_mahalanobis)
        
        return np.argmax(self.g) + 1