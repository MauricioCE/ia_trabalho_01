# models_mqo.py
# MQO (Least Squares) One-Vs-Rest para classificação — NumPy puro

import numpy as np

class MQO:
    """
    Classificador linear via MQO em esquema One-Vs-Rest.
    - X: (N, p)
    - y: (N,), rótulos inteiros em {1..C}
    - B: (p+1, C) se add_intercept=True, senão (p, C)
    Decisão por argmax dos scores lineares.
    """
    def __init__(self, add_intercept: bool = True, C: int | None = None):
        self.add_intercept = add_intercept
        self.B = None
        self.C = C
        self.p = None

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        Y = np.zeros((y.size, self.C))
        Y[np.arange(y.size), y - 1] = 1.0
        return Y

    def _augment(self, X: np.ndarray) -> np.ndarray:
        if not self.add_intercept:
            return X
        N = X.shape[0]
        return np.hstack([np.ones((N, 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta B pela solução fechada de MQO:
        B = (X^T X)^(-1) X^T Y   (usando pinv para estabilidade numérica)
        """
        self.p = X.shape[1]

        Xaug = self._augment(X)    # (N, p+1) se intercepto
        Y    = self._one_hot(y)    # (N, C)

        # pseudo-inversa para estabilidade
        self.B = np.linalg.pinv(Xaug.T @ Xaug) @ (Xaug.T @ Y)  # (p+1, C)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna scores lineares: S = [1 X] B  -> (N, C)
        """
        Xaug = self._augment(X)
        return Xaug @ self.B

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predição de rótulos em {1..C} via argmax dos scores.
        """
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1) + 1
