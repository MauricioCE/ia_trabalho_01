import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from _mauricio import MQO_m
from helpers.q2_helper import calculate_accuracy, grafico_dispersao_inicial
from modelos.gaussian_classifiers import GaussianClassifier
from modelos.mqo import MQO

# *********************************************************** #
#                           DADOS                             #
# *********************************************************** #

dados = np.loadtxt('dados/EMGsDataset.csv', delimiter=',') # Linhas: Sensores, Colunas: Amostras
dados = dados.T  # Linhas: Amostras, Colunas: Sensores

X = dados[:, :-1]
y = dados[:, -1:].astype(int).flatten()
y_unicos = np.unique(y)

N = X.shape[0] # quantidade de observações
p = X.shape[1] # quantidade de variáveis regressoras
C = len(np.unique(y)) # quantidade de classes

# X e y para o modelo MQO
X_mqo_features = X # Matriz com as observações das variáveis regressoras
Y_mqo = y # Matriz

# X e y para os modelos Gaussianos
X_gauss = X.T
Y_gauss = Y_mqo.T

# Inicializar listas para armazenar as acurácias
acuracias_MQO = [] # Mínimos Quadrados Ordinários
acuracias_CGT = [] # Classificador Gaussiano Tradicional
acuracias_CGC = [] # Classificador Gaussiano com Covariâncias Iguais
acuracias_CGM = [] # Classificador Gaussiano com Matriz Agregada
acuracias_CGR = [] # Classificador Gaussiano com Matriz Regularizado
acuracias_CGB = [] # Classificador Gaussiano de Bayes Ingenuo

# *********************************************************** #
#                     TREINAMENTO E TESTE                     #
# *********************************************************** #

contador_progresso = 1
interacoes = 500
for count in range(interacoes):
    # ***********  Aleatorizando os dados  *********** #
    
    idx = np.random.permutation(N)
    X_rodada = X_mqo_features[idx,:] # X embaralhado
    Y_rodada = Y_mqo[idx] # Y embaralhado

    X_treino = X_rodada[:int(N*.8),:] # 80% para treino
    y_treino = Y_rodada[:int(N*.8)] # 80% para treino

    X_teste = X_rodada[int(N*.8):,:] # 20% para teste
    y_teste_sample = X_rodada[int(0.8 * X.shape[0]), :].reshape(1,2) # Prof
    y_teste = Y_rodada[int(N*.8):] # 20% para teste
    
    # ***********  Instanciar e treinar os modelos  *********** #

    modelo_MQO = MQO(add_intercept = True, C = C)
    # modelo_CGT = GaussianClassifier(features_treino.T, classes_treino.T)

    modelo_MQO.fit(X_treino, y_treino)
    # modelo_CGT.fit()

    # ***********  Predições  *********** #

    predicao_MQO = modelo_MQO.predict(X_teste)
    # predicao_CGT = modelo_CGT.predict(features_teste_sample)

    # ***********  Acurácias  *********** #
    
    acuracias_MQO.append(np.mean(predicao_MQO == y_teste))
    # acuracias_CGT.append(calculate_accuracy(classes_teste.T, predicao_CGT))

    # ***********  Contador de progresso  *********** #

    if(contador_progresso % 50 == 0):
        print(f"{contador_progresso}/500")
    contador_progresso = contador_progresso + 1

# *********************************************************** #
#                            PRINTS                           #
# *********************************************************** #

media_mqo = np.mean(acuracias_MQO)
# media_CGT = np.mean(acuracias_CGT)

print(f"MQO: {media_mqo:.4f}")
# print(f"CGT: {media_CGT}")

# Gráfico de dispersão dos dados
# grafico_dispersao_inicial(dados, classe_ids)



