import numpy as np

class GaussTradicional:
    def __init__(self, X_train, y_train, lam: float = 0.0):
        # Identifica as classes únicas no conjunto de treinamento

        self.lam = float(lam)  # << Friedman shrinkage (0.0 = tradicional)

        self.classes = np.unique(y_train)
        self.C = len(self.classes)  # Número de classes
        self.p, self.N = X_train.shape  # p = número de atributos, N = número de amostras

        # Separa os dados por classe (assume que y_train é uma matriz 2D)
        self.X = [X_train[:, y_train[0] == i] for i in self.classes]

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


            Sigma_raw = np.cov(self.X[i])

            # Friedman: Σ(λ) = (1-λ) Σ + λ diag(Σ)
            D = np.diag(np.diag(Sigma_raw))
            Sigma_shr = (1.0 - self.lam) * Sigma_raw + self.lam * D

            # sua regularização e cálculos continuam iguais a seguir:
            self.Sigma[i] = Sigma_shr + 1e-6 * np.eye(self.p)

            
            
            sign, logdet = np.linalg.slogdet(self.Sigma[i])
            if sign <= 0:
                self.Sigma[i] = self.Sigma[i] + 1e-6 * np.eye(self.p)
                sign, logdet = np.linalg.slogdet(self.Sigma[i])
            self.Sigma_det[i] = logdet   # << agora guardamos o LOG-DET
            self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])

            # Probabilidade a priori da classe i (frequência relativa)
            self.P[i] = self.n[i] / self.N

    def predict(self, x_test):
        # Calcula a função discriminante para cada classe
        for i in range(self.C):
            # Distância de Mahalanobis entre x_test e a média da classe i
            d_mahalanobis = ((x_test - self.mu[i]).T @ self.Sigma_inv[i] @ (x_test - self.mu[i]))[0, 0]

            # Função discriminante baseada na probabilidade a priori, covariância e distância
            self.g[i] = np.log(self.P[i]) - 0.5 * self.Sigma_det[i] - 0.5 * d_mahalanobis

        # Retorna o índice da classe com maior g[i], somado de +1 para ajustar à numeração das classes (começando em 1)
        a = np.argmax(self.g) + 1   
        return a 
    
    def predict_batch(self, X_test):
        """
        Predição vetorizada para um lote de amostras.
        Espera X_test com shape (p, N_te), igual ao padrão dessa classe.
        Retorna rótulos em {1..C} com shape (N_te,).
        """


        p, N_te = X_test.shape
        C = self.C

        # empilha parâmetros por classe
        means = np.hstack(self.mu).T            # (C, p) - cada mu[i] é (p,1)
        invs  = np.stack(self.Sigma_inv, 0)     # (C, p, p)

        # ATENÇÃO: aqui assumo que você já salvou LOG-determinantes em self.Sigma_det
        # (como te mostrei antes com slogdet). Se ainda guarda o determinante "cru",
        # ajuste seu fit pra guardar o logdet.
        logdet = np.array(self.Sigma_det)       # (C,)
        logpri = np.log(np.array(self.P) + 1e-12)  # (C,)

        # diferenças (N_te, C, p)
        X = X_test.T                            # (N_te, p)
        diff = X[:, None, :] - means[None, :, :]

        # distância de Mahalanobis em lote -> (N_te, C)
        maha = np.einsum('ncp,cpq,ncq->nc', diff, invs, diff)

        # função discriminante log
        scores = logpri[None, :] - 0.5 * (p*np.log(2.0*np.pi) + logdet[None, :] + maha)

        return np.argmax(scores, axis=1) + 1
