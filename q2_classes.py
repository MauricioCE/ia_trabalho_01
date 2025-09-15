import numpy as np

class MQO:
    def __init__(self):
        self.betas = {}

    # Q X de treinamento tem que ser SEM interceptor, pois estou colandos a coluna de 1s aqui
    # O y tbm tem que ser de treino
    def fit(self, X_treino, y_treino):
        X_treino_interceptor = np.hstack([np.ones((X_treino.shape[0], 1)), X_treino])
        classes = np.unique(y_treino)

        for classe in classes:
            y_binario = (y_treino == classe).astype(int)
            self.betas[classe] = np.linalg.inv(X_treino_interceptor.T @ X_treino_interceptor) @ X_treino_interceptor.T @ y_binario
            
    def predict(self, X_teste):
        X_interceptor = np.hstack([np.ones((X_teste.shape[0], 1)), X_teste])
        scores = np.zeros((X_teste.shape[0], len(self.betas)))
        for i, c in enumerate(self.betas):
            scores[:, i] = X_interceptor @ self.betas[c]
        return np.argmax(scores, axis=1) + 1


