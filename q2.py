import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from _mauricio.q2_classes import MQO, GaussianClassifier
from helpers.q2_helper import calculate_accuracy, grafico_dispersao_inicial

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
Y_MQO_classes_one_shot = np.eye(C)[classes - 1] # Matriz one-hot no formato [x1, x2, x3, x4, x5] (5 classes)

X_Gauss_features = features.T
Y_Gauss_classes_one_shot = Y_MQO_classes_one_shot.T

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
    Y_rodada = Y_MQO_classes_one_shot[idx,:] # Y embaralhado. Já é one-shot

    features_treino = X_rodada[:int(N*.8),:] # 80% para treino
    classes_treino = Y_rodada[:int(N*.8),:] # 80% para treino. Já é one-shot

    features_teste = X_rodada[int(N*.8):,:] # 20% para teste
    features_teste_sample = X_rodada[int(0.8 * features.shape[0]), :].reshape(1,2) # Prof
    classes_teste = Y_rodada[int(N*.8):,:] # 20% para teste. Já é one-shot
    
    # 1. Instanciar e treinar os modelos
    modelo_MQO = MQO()
    modelo_CGT = GaussianClassifier(features_treino.T, classes_treino.T)

    modelo_MQO.fit(features_treino, classes_treino, classes_unicas)
    modelo_CGT.fit()

    # 2. Fazer previsões
    predicao_MQO = modelo_MQO.predict(features_teste, len(classes_unicas))
    predicao_CGT = modelo_CGT.predict(features_teste_sample)

    # 3. Calcular e armazenar as acurácias
    indices_classes_teste = np.argmax(classes_teste, axis=1) + 1
    acuracias_MQO.append(calculate_accuracy(indices_classes_teste, predicao_MQO))
    acuracias_CGT.append(calculate_accuracy(classes_teste.T, predicao_CGT))

media_mqo = np.mean(acuracias_MQO)
media_CGT = np.mean(acuracias_CGT)

print(f"MQO: {media_mqo}")
print(f"CGT: {media_CGT}")

# Gráfico de dispersão dos dados
# grafico_dispersao_inicial(dados, classe_ids)


bp=1


