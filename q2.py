import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from q2_classes import MQO
from q2_helper import calculate_accuracy, grafico_dispersao_inicial

# >>>>>>>>>>>>  DADOS  <<<<<<<<<<<<

# Carregar os dados
dados = np.loadtxt('EMGsDataset.csv', delimiter=',') # Linhas: Sensores, Colunas: Amostras
dados = dados.T  # Linhas: Amostras, Colunas: Sensores

features = dados[:, :-1]
classes = dados[:, -1:].astype(int).flatten()
classes_unicas = np.unique(classes)

N = features.shape[0] # quantidade de observações
p = features.shape[1] # quantidade de variáveis regressoras
C = len(np.unique(classes)) # quantidade de classes

X_MQO_features = features # Matriz com as observações das variáveis regressoras
Y_MQO_classes = np.eye(C)[classes - 1] # Matriz one-hot no formato [x1, x2, x3, x4, x5] (5 classes)

X_Gauss_features = features.T
Y_Gauss_classes = Y_MQO_classes.T

# Inicializar listas para armazenar as acurácias
acuracias_MQO = [] # Mínimos Quadrados Ordinários
acuracias_CGT = [] # Classificador Gaussiano Tradicional
acuracias_CGC = [] # Classificador Gaussiano com Covariâncias Iguais
acuracias_CGM = [] # Classificador Gaussiano com Matriz Agregada
acuracias_CGR = [] # Classificador Gaussiano com Matriz Regularizado
acuracias_CGB = [] # Classificador Gaussiano de Bayes Ingenuo

for i in range(500):
    idx = np.random.permutation(N)

    X_rodada = X_MQO_features[idx,:] # X  embaralhado
    Y_rodada = Y_MQO_classes[idx,:] # Y embaralhado

    features_treino = X_rodada[:int(N*.8),:] # 80% para treino
    classes_treino = Y_rodada[:int(N*.8),:] # 80% para treino

    features_teste = X_rodada[int(N*.8):,:] # 20% para teste
    classes_teste = Y_rodada[int(N*.8):,:] # 20% para teste
    
    # 1. Instanciar e treinar os modelos
    modelo_MQO = MQO()
    modelo_MQO.fit(features_treino, classes_treino)

    bp=1
    
    # 2. Fazer previsões
    predicao_MQO = modelo_MQO.predict(features_teste)

    # 3. Calcular e armazenar as acurácias
    acuracias_MQO.append(calculate_accuracy(classes_teste, predicao_MQO))
    

# Gráfico de dispersão dos dados
# grafico_dispersao_inicial(dados, classe_ids)


bp=1


