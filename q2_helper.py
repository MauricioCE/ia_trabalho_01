import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)