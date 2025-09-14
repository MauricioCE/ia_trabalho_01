import numpy as np
from numpy.linalg import pinv
import pandas as pd
from helper_q1 import plotar_grafico_1, gerar_matrizes, predicao, plotar_modelo_regressao, imprimir_tabela_resultados, RSS, exportar_dados_para_csv

# >>>>>>>>>>>>  DADOS  <<<<<<<<<<<<

# Carregar os dados
dados = np.loadtxt('aerogerador.dat', delimiter='\t')

# Transpor os dados
dados_t = dados.T

# Gerar a Matriz X e o vetor y
X_sem_intercepor, y = gerar_matrizes(dados)   

# Plotando o gráfico de dispersão
# plotar_grafico_1(X, y)

# Pegando os tamanhos de N e y
N,p = X_sem_intercepor.shape

# Adicionando 1s à primera coluna de X (interceptador)
X_com_interceptor = np.c_[np.ones(X_sem_intercepor.shape[0]), X_sem_intercepor]

# Dados para o MQO Regularizado
valores_lambda = [0, 0.25, 0.5, 0.75, 1]
matriz_identidade = np.identity(X_com_interceptor.shape[1])

# Resultados do RSS
rss_media = []
rss_mqo_sem_interceptor = []
rss_mqo_regularizado_0 = []
rss_mqo_regularizado_25 = []
rss_mqo_regularizado_50 = []
rss_mqo_regularizado_75 = []
rss_mqo_regularizado_100 = []


# >>>>>>>>>>>>  TREINAMENTO E TESTE  <<<<<<<<<<<<

for i in range(500):

    ###############################################################
    #                      Gerando os dados                       #
    ###############################################################

    # Embaralhar os dados
    idx = np.random.permutation(N)

    # Sem intercepto
    X_rodada_sem_interceptor = X_sem_intercepor[idx,:] # X embaralhado
    X_treino_sem_interceptor = X_rodada_sem_interceptor[:int(N*.8),:] # 80% para treino
    X_teste_sem_interceptor = X_rodada_sem_interceptor[int(N*.8):,:] # 20% para teste

    # Com intercepto
    X_rodada_com_interceptor = X_com_interceptor[idx,:] # X_interceptor  embaralhado
    X_treino_com_interceptor = X_rodada_com_interceptor[:int(N*.8),:] # 80% para treino
    X_teste_com_interceptor = X_rodada_com_interceptor[int(N*.8):,:] # 20% para teste
    
    # y
    y_rodada = y[idx,:] # y embaralhado
    y_treino = y_rodada[:int(N*.8),:] # 80% para treino
    y_teste = y_rodada[int(N*.8):,:] # 20% para teste

    ###############################################################
    #                   Treinamento dos modelos                   #
    ###############################################################

    # Modelo MQO Regularizado
    betas_hat_MQO_regularizado = {}

    for valor in valores_lambda:
        termo_regularizacao = valor * matriz_identidade
        termo_regularizacao[0, 0] = 0 # Não regulariza o intercepto

        beta_hat_MQO_interceptor = pinv(X_treino_com_interceptor.T @ X_treino_com_interceptor + termo_regularizacao) @ X_treino_com_interceptor.T @ y_treino

        betas_hat_MQO_regularizado[valor] = beta_hat_MQO_interceptor

    # Modelo baseado MQO (sem intercepto)
    beta_hat_MQO_sem_interceptor = np.linalg.pinv(X_treino_sem_interceptor.T @ X_treino_sem_interceptor) @ X_treino_sem_interceptor.T @ y_treino
    beta_hat_MQO_sem_interceptor = np.vstack((
        np.zeros((1,1)),beta_hat_MQO_sem_interceptor
    ))

    # Modelo baseado na média
    beta_hat_MEDIA = np.array([ [np.mean(y_treino)], [0] ])

    ###############################################################
    #                     Teste de desempenho                     #
    ###############################################################

    y_pred = predicao(X_teste_com_interceptor, beta_hat_MQO_sem_interceptor)
    rss_mqo_sem_interceptor.append(RSS(y_teste, y_pred))

    y_pred = predicao(X_teste_com_interceptor, beta_hat_MEDIA)
    rss_media.append(RSS(y_teste, y_pred))

    for valor_lambda, beta in betas_hat_MQO_regularizado.items():
        y_pred = predicao(X_teste_com_interceptor, beta)
        rss = RSS(y_teste, y_pred)

        if valor_lambda == 0:
            rss_mqo_regularizado_0.append(rss)
        elif valor_lambda == 0.25:
            rss_mqo_regularizado_25.append(rss)
        elif valor_lambda == 0.5:
            rss_mqo_regularizado_50.append(rss)
        elif valor_lambda == 0.75:
            rss_mqo_regularizado_75.append(rss)
        elif valor_lambda == 1:
            rss_mqo_regularizado_100.append(rss)

    ###############################################################
    #                          Gráficos                          #
    ###############################################################

    # plotar_modelo_regressao(X_treino_sem_interceptor, y_treino, beta_hat_MQO_sem_interceptor, "linear", "MQO sem interceptor")
    
    # plotar_modelo_regressao(X_treino_sem_interceptor, y_treino, beta_hat_MEDIA[0], "media", "Modelo média")

    # for valor_lambda, beta in betas_hat_MQO_regularizado.items():
    #     nome_modelo = f"MQO regularizado (lambda={valor_lambda})"
    #     plotar_modelo_regressao(X_treino_sem_interceptor, y_treino, beta, "linear", nome_modelo)



# >>>>>>>>>>>>  CALCULANDO AS ESTATÍSTICAS  <<<<<<<<<<<<

resultados_para_tabela = {
    "Média da variável dependente": {
        "media": np.mean(rss_media),
        "desvio_padrao": np.std(rss_media),
        "maior_valor": np.max(rss_media),
        "menor_valor": np.min(rss_media)
    },
    "MQO sem intercepto": {
        "media": np.mean(rss_mqo_sem_interceptor),
        "desvio_padrao": np.std(rss_mqo_sem_interceptor),
        "maior_valor": np.max(rss_mqo_sem_interceptor),
        "menor_valor": np.min(rss_mqo_sem_interceptor) 
    },
    "MQO tradicional": {
        "media": np.mean(rss_mqo_regularizado_0),
        "desvio_padrao": np.std(rss_mqo_regularizado_0),
        "maior_valor": np.max(rss_mqo_regularizado_0),
        "menor_valor": np.min(rss_mqo_regularizado_0)
    },
    "MQO regularizado (0.025)": {
        "media": np.mean(rss_mqo_regularizado_25),
        "desvio_padrao": np.std(rss_mqo_regularizado_25),
        "maior_valor": np.max(rss_mqo_regularizado_25),
        "menor_valor": np.min(rss_mqo_regularizado_25)
    },
    "MQO regularizado (0.05)": {
        "media": np.mean(rss_mqo_regularizado_50),
        "desvio_padrao": np.std(rss_mqo_regularizado_50),
        "maior_valor": np.max(rss_mqo_regularizado_50),
        "menor_valor": np.min(rss_mqo_regularizado_50)
    },
    "MQO regularizado (0.075)": {
        "media": np.mean(rss_mqo_regularizado_75),
        "desvio_padrao": np.std(rss_mqo_regularizado_75),
        "maior_valor": np.max(rss_mqo_regularizado_75),
        "menor_valor": np.min(rss_mqo_regularizado_75)
    },
    "MQO regularizado (1)": {
        "media": np.mean(rss_mqo_regularizado_100),
        "desvio_padrao": np.std(rss_mqo_regularizado_100),
        "maior_valor": np.max(rss_mqo_regularizado_100),
        "menor_valor": np.min(rss_mqo_regularizado_100)
    },
}


# >>>>>>>>>>>>  PRINTANDO OS RESULTADOS  <<<<<<<<<<<<

imprimir_tabela_resultados(resultados_para_tabela)
exportar_dados_para_csv(resultados_para_tabela)

