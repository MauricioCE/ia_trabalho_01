import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

def grafico_dispersao_inicial(dados, classe_ids):
    nomes_das_emocoes = ["Neutro", "Sorriso", "Sobrancelhas Levantadas", "Surpreso", "Rabugento"]
    cores_por_emocao = ["red", "blue", "green", "yellow", "cyan"]

    # Loop para plotar os dados de cada classe com cor e legenda apropriadas
    for indice, classe in enumerate(classe_ids):
        amostras_da_classe = dados[:, dados[-1, :] == classe][0:-1, :].T
        plt.scatter(amostras_da_classe[:, 0], amostras_da_classe[:, 1],
                    c=cores_por_emocao[indice],
                    label=nomes_das_emocoes[indice],
                    edgecolors='k')

    plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
    plt.ylabel("Sensor 2 (Zigomático Maior)")
    plt.legend()
    plt.show()

def calculate_accuracy(y_teste, y_pred):
    return np.mean(y_teste == y_pred)

# Tentativa do Gemini. Não gostei
def plotar_regioes_decisao_dinamico(model, X, y, title):
    unique_classes = np.unique(y)
    cores_regioes = ['black','orange', 'purple', 'pink', 'brown']
    marcadores = ['o', 'o', 'o', 'o', 'o', 'o', 'o']
    
    # Cria a grade de pontos para a visualização
    x_min, x_max = X[:, 0].min() - 20, X[:, 0].max() + 20
    y_min, y_max = X[:, 1].min() - 20, X[:, 1].max() + 20

    # Aumenta o passo para reduzir o número de pontos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 20), np.arange(y_min, y_max, 20))

    # Faz as predições para todos os pontos da grade
    a = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(a)
    Z = Z.reshape(xx.shape)

    # Cria uma única tela de plotagem
    plt.figure(figsize=(5, 4))
    
    # Plota as regiões de decisão PRIMEIRO
    

    # Plota os pontos de dados originais POR CIMA das regiões
    for i, classe in enumerate(unique_classes):
        plt.scatter(X[y == classe, 0], X[y == classe, 1],
                    c=cores_regioes[i],
                    label=f'Classe {classe}',
                    marker=marcadores[i],
                    edgecolors='k', s=80)
        
    plt.contourf(xx, yy, Z, alpha=0.7, cmap=ListedColormap(cores_regioes))
    
    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.title(title)
    plt.legend()
    plt.show()