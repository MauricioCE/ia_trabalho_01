import numpy as np

class MQO:
    def __init__(self):
        self.betas = {}

    # Q X de treinamento tem que ser SEM interceptor, pois estou colocando a coluna de 1s aqui
    # O y tem que ser de y_treino_one_shot -> ex: [1, 0, 0, 0, 0]
    def fit(self, X_treino, y_treino_one_shot, classes_unicas):
        X_treino_interceptor = np.hstack([np.ones((X_treino.shape[0], 1)), X_treino])

        for index, classe in enumerate(classes_unicas):
            # Valor da classe no treino. Nx1
            y_valor_classe = y_treino_one_shot[:, index]
            # Vai popular o diciónario de betas para uma dada classe. O valor da classe é a chave
            # O valor é um array de 1x3, [B0, B1, B2], onde o B0 é do interceptor, B1 e B2 dos sensores
            self.betas[classe] = np.linalg.inv(X_treino_interceptor.T @ X_treino_interceptor) @ X_treino_interceptor.T @ y_valor_classe

        bp=1 # TODO: Breakpoint para teste. Remover depois
            
    def predict(self, X_teste, quant_classes):
        # Tentei 3 estratégias:
        # Maior nota da soma: Baixíssima acurácia
        # Mediana das notas: Todas as acurácias tiveram valor de 0.2
        # Por observação de vezez: Se saiu melhor, aparentemente

        # Criando o interceptor
        X_interceptor = np.hstack([np.ones((X_teste.shape[0], 1)), X_teste])

        # Matriz de notas zeradas. Detalhe, as notas de cada classe ficaram nas suas respectivas colunas
        # Tipo, as notas da classe 1 ficarão na coluna 1, e assim por diante
        notas = np.zeros((X_teste.shape[0], len(self.betas)))
        # notas = np.zeros(quant_classes, dtype=float)

        # Calculando as notas para cada classe
        for index, chave_beta_hat in enumerate(self.betas):
            betas_classe = self.betas[chave_beta_hat]
            notas_classe = X_interceptor @ betas_classe
            notas[:, index] = notas_classe

        # Retornando a classe com maior nota
        # TODO: Ver se vai precisar ajudar com +1
        # TODO: remover classe_predita e retornar direto
        # Retorna uma array com os índices das classes que foram melhor para cada amostra.
        # É por isso que é um array de 10.000 índices, pois são 10.000 observações
        # O axi=1 garante que está se pegando o maior valor de cada linha, olha os valores de todas as colunas
        
        classe_predita = np.argmax(notas, axis=1) + 1
        return classe_predita