import numpy as np

class GaussianClassifier:
    def __init__(self, X_train, y_train):
        # Identifica as classes únicas no conjunto de treinamento
        self.classes = np.unique(y_train)
        self.C = len(self.classes)  # Número de classes
        self.p, self.N = X_train.shape  # p = número de atributos, N = número de amostras

        # Separa os dados por classe (assume que y_train é uma matriz 2D)
        self.X = [X_train[:, y_train[0, :] == i] for i in self.classes]

        # Conta quantas amostras há em cada classe
        self.n = [Xi.shape[1] for Xi in self.X]

        # Inicializa listas para armazenar parâmetros estatísticos de cada classe
        self.Sigma = [None] * self.C         # Matrizes de covariância
        self.Sigma_det = [None] * self.C     # Determinantes das covariâncias
        self.Sigma_inv = [None] * self.C     # Inversas das covariâncias
        self.mu = [None] * self.C            # Vetores de média
        self.P = [None] * self.C             # Probabilidades a priori (frequência relativa)

        # Inicializa vetor de funções discriminantes
        self.g = [None] * self.C

    def fit(self):
        # Calcula os parâmetros estatísticos para cada classe
        for i in range(self.C):
            # Média dos dados da classe i
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)

            # Covariância dos dados da classe i
            self.Sigma[i] = np.cov(self.X[i])

            # Determinante da matriz de covariância
            self.Sigma_det[i] = np.linalg.det(self.Sigma[i])

            # Inversa da matriz de covariância (usando pseudo-inversa para estabilidade numérica)
            self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])

            # Probabilidade a priori da classe i (frequência relativa)
            self.P[i] = self.n[i] / self.N

    def predict(self, x_test):
        # Calcula a função discriminante para cada classe
        for i in range(self.C):
            # Distância de Mahalanobis entre x_test e a média da classe i
            d_mahalanobis = ((x_test - self.mu[i]).T @ self.Sigma_inv[i] @ (x_test - self.mu[i]))[0, 0]

            # Função discriminante baseada na probabilidade a priori, covariância e distância
            self.g[i] = np.log(self.P[i]) - 0.5 * np.log(self.Sigma_det[i]) - 0.5 * d_mahalanobis

        # Retorna o índice da classe com maior g[i], somado de +1 para ajustar à numeração das classes (começando em 1)
        a = np.argmax(self.g) + 1   
        return a 