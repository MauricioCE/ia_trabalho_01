import numpy as np
import matplotlib.pyplot as plt
from helpers.q2_helper import calculate_accuracy, grafico_dispersao_inicial
from modelos.gauss_cov_agregrada import GaussCovarianciaAgregada
from modelos.gauss_covariancia import GaussCovarianciasGlobal
from modelos.mqo import MQO
from modelos.gauss_tradicional import GaussTradicional
from modelos.gauss_naive import GaussNaive

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
acuracias_G_T = [] # Classificador Gaussiano Tradicional
acuracias_G_C_G = [] # Classificador Gaussiano com Covariâncias Global
acuracias_G_M_A= [] # Classificador Gaussiano com Matriz Agregada
acuracias_G_M_R = [] # Classificador Gaussiano com Matriz Regularizado
acuracias_G_B_N = [] # Classificador Gaussiano de Bayes Naive
# *********************************************************** #
#                     TREINAMENTO E TESTE                     #
# *********************************************************** #

contador_progresso = 0
interacoes = 500
for count in range(interacoes):
    # ***********  Aleatorizando os dados  *********** #
    
    idx = np.random.permutation(N)
    X_rodada = X_mqo_features[idx,:] # X embaralhado
    Y_rodada = Y_mqo[idx] # Y embaralhado

    X_treino = X_rodada[:int(N*.8),:] # 80% para treino
    y_treino = Y_rodada[:int(N*.8)] # 80% para treino

    X_teste = X_rodada[int(N*.8):,:] # 20% para teste
    y_teste = Y_rodada[int(N*.8):] # 20% para teste
    X_teste_sample = X_rodada[int(0.8 * X.shape[0]), :].reshape(1,2) # Prof
    
    # ***********  Instanciar e treinar os modelos  *********** #

    modelo_MQO = MQO(add_intercept = True, C = C)
    modelo_G_T = GaussTradicional(X_treino.T, y_treino.T.reshape(1,len(y_treino)))
    modelo_G_C_G = GaussCovarianciasGlobal(X_treino.T, y_treino.T.reshape(1,len(y_treino)))
    modelo_G_B_N = GaussNaive(X_treino.T, y_treino.T.reshape(1,len(y_treino)))
    modelo_G_M_A = GaussCovarianciaAgregada(X_treino.T, y_treino.T.reshape(1,len(y_treino)))

    modelo_MQO.fit(X_treino, y_treino)
    modelo_G_T.fit()
    modelo_G_C_G.fit()
    modelo_G_B_N.fit()
    modelo_G_M_A.fit()

    # ***********  Predições  *********** #

    predicao_MQO = modelo_MQO.predict(X_teste)
    predicao_G_T = modelo_G_T.predict(X_teste_sample)
    predicao_G_C_G = modelo_G_C_G.predict(X_teste_sample)
    predicao_G_B_N = modelo_G_B_N.predict(X_teste_sample)
    predicao_G_M_A = modelo_G_M_A.predict(X_teste_sample)

    # ***********  Acurácias  *********** #
    
    acuracias_MQO.append(np.mean(predicao_MQO == y_teste))
    acuracias_G_T.append(calculate_accuracy(y_teste.T, predicao_G_T))
    acuracias_G_C_G.append(calculate_accuracy(y_teste.T, predicao_G_C_G))
    acuracias_G_B_N.append(calculate_accuracy(y_teste.T, predicao_G_B_N))
    acuracias_G_M_A.append(calculate_accuracy(y_teste.T, predicao_G_M_A))
    
    # ***********  Contador de progresso  *********** #

    if(contador_progresso  % 50 == 0):
        print(f"{contador_progresso}/500")
    contador_progresso = contador_progresso + 1

print(f"{contador_progresso}/500\nFinalizado\n")

# *********************************************************** #
#                            PRINTS                           #
# *********************************************************** #

media_MQO = np.mean(acuracias_MQO)
media_G_T = np.mean(acuracias_G_T)
media_G_C_G = np.mean(acuracias_G_C_G)
media_G_B_N = np.mean(acuracias_G_B_N)
media_G_M_A = np.mean(acuracias_G_M_A)

print(f"MQO: {media_MQO:.4f}")
print(f"Gauss Tradicional: {media_G_T}")
print(f"Gauss Covariância Global: {media_G_C_G}")
print(f"Gauss Naive: {media_G_B_N}")
print(f"Gauss Matriz Agregada: {media_G_M_A}")

# Gráfico de dispersão dos dados
# grafico_dispersao_inicial(dados, classe_ids)



