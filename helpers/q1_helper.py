import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Fazer a predição
def predicao(X,beta):
    return X @ beta

def RSS(y_teste, y_pred):
    return np.sum((y_teste - y_pred)**2)

# Gerar a matriz X e o vetor y
def gerar_matrizes(dados):
    X = dados[:, :-1]
    y = dados[:, -1:]
    return X, y

# 
def plotar_grafico_1(X, y):
    velocidade = X
    potencia = y

    plt.scatter(velocidade, potencia,
                c='r',
                edgecolors='k',
                label='Potência')

    plt.xlabel("Velocidade do vento")
    plt.ylabel("Potência gerada")
    plt.title("Curva de Potência do Aerogerador")
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.2)
    plt.xlim(velocidade.min() - 1, velocidade.max() + 1)
    plt.ylim(potencia.min() - 30, potencia.max() + 50)
    plt.show()

# Função para plotar o gráfico de dispersão com a linha de regressão
def plotar_modelo_regressao(X, y, beta, tipo_modelo, nome_modelo, grau=None):
   
    # Plota os dados observados
    plt.scatter(X, y, c='r',
                edgecolors='k',
                label='Potência')

    # Plota a linha de previsão com base no tipo de modelo
    if tipo_modelo == 'linear':
        x_range = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
        x_range_com_intercepto = np.c_[np.ones(x_range.shape[0]), x_range]
        y_pred = x_range_com_intercepto @ beta
        plt.plot(x_range, y_pred, color='blue', linewidth=2, label='Linha de Regressão')
    
    elif tipo_modelo == 'media':
        plt.axhline(y=beta, color='green', linestyle='--', linewidth=2, label='Modelo da Média')
    elif tipo_modelo == 'polinomial':
        x_range = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
        from helpers.q1_helper import poly_design  # se estiver em outro módulo
        Phi = poly_design(x_range, grau if grau is not None else (beta.shape[0]-1))
        y_pred = Phi @ beta
        plt.plot(x_range, y_pred, color='purple', linewidth=2, label=f'Polinomial (d={grau})')
            
    # Adiciona títulos e legendas
    plt.xlabel("Velocidade do vento")
    plt.ylabel("Potência gerada")
    plt.title("Curva de Potência do Aerogerador")
    plt.suptitle(nome_modelo, fontsize=16)
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.2)
    plt.show()

# 
def imprimir_tabela_resultados(resultados_dic):
    # Defina a largura das colunas para alinhamento
    col_width = 28
    num_width = 15
    line = '-' * (col_width + 4 * num_width + 5)
    
    # Imprime o cabeçalho da tabela
    print(line)
    print(f"| {'Modelos':<{col_width}} | {'Média':<{num_width}} | {'Desvio-Padrão':<{num_width}} | {'Maior Valor':<{num_width}} | {'Menor Valor':<{num_width}} |")
    print(line)

    # Imprime os resultados de cada modelo, iterando sobre o dicionário
    for modelo, dados in resultados_dic.items():
        media = dados['media']
        desvio_padrao = dados['desvio_padrao']
        maior_valor = dados['maior_valor']
        menor_valor = dados['menor_valor']
        
        print(f"| {modelo:<{col_width}} | {media:>{num_width}.2f} | {desvio_padrao:>{num_width}.2f} | {maior_valor:>{num_width}.2f} | {menor_valor:>{num_width}.2f} |")
    
    print(line)

# Não consegui exportar pelo panda, então estou exportando para um csv e fazendo os gráficos no excel
def exportar_dados_para_csv(resultados):
    os.makedirs('dados', exist_ok=True)
    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
    df_resultados.index.name = 'Modelo'
    caminho_arquivo = os.path.join('dados', 'resultados_q1.csv')
    df_resultados.to_csv(caminho_arquivo, index=True, sep=';', decimal=',')

    print("Arquivo 'resultados.csv' criado com sucesso na pasta 'dados'.")



# --- Design polinomial Φ = [1, x, x^2, ..., x^d] ---
def poly_design(x, d):
    x = x.reshape(-1, 1)
    Phi = np.hstack([np.ones((x.shape[0], 1))] + [x**k for k in range(1, d+1)])
    return Phi

# --- Ajuste por MQO/Tikhonov (não regulariza o intercepto) ---
def fit_ridge(Phi, y, lam=0.0):
    M = Phi.shape[1]
    D = np.eye(M); D[0, 0] = 0.0
    return np.linalg.pinv(Phi.T @ Phi + lam * D) @ (Phi.T @ y)