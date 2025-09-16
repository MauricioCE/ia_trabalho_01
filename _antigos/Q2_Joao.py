# ====================================================
# Tarefa de Classificação (EMG) — ESQUELETO INICIAL
# Somente NumPy + Matplotlib. Apenas lê os dados.
# Vamos preencher as etapas aos poucos.
# ====================================================

import numpy as np
import matplotlib.pyplot as plt

from modelos.gauss_tradicional import GaussTradicional
from modelos.mqo import MQO

# ----------------------------------------------------
# Configurações básicas
# ----------------------------------------------------
DATA_PATH = "dados/EMGsDataset.csv"   # ajuste se necessário
SHOW_PLOTS = False              # por enquanto, nada de gráficos

def one_hot_from_labels(labels, C):
    """
    labels: vetor (N,) com valores inteiros de 1..C
    Retorna: matriz (N,C) com one-hot
    """
    Y = np.zeros((labels.size, C))
    Y[np.arange(labels.size), labels-1] = 1.0
    return Y

# ----------------------------------------------------
# main — apenas leitura e preparo dos dados
# ----------------------------------------------------
def main():

    # ------------------------------------------------
    # ETAPA 1 - ORGANIZAÇÃO DOS DADOS
    # ------------------------------------------------

    #Leitura do CSV (aceita 3xN ou N x 3)

    data = np.loadtxt(DATA_PATH, delimiter=",")

    if data.shape[0] == 3 and data.shape[1] != 3:
        data = data.T  # 3xN -> N x 3

    #Separa X e y
    X_all = data[:, :2].astype(float)  # (N, 2)
    y_all = data[:, 2].astype(int)     # (N,), rótulos 1..5

    #Metadados úteis
    N, p = X_all.shape
    classes = np.unique(y_all)
    C = classes.size

    #prepara variáveis no formato pedido
    # --- para MQO:
    X_MQO = X_all                          # (N,p)
    Y_MQO = one_hot_from_labels(y_all, C)  # (N,C)

    # --- para Bayesianos:
    X_BAYES = X_all.T        # (p,N)
    Y_BAYES = Y_MQO.T        # (C,N)

    # 4) Resumo rápido
    print("==============================================")
    print("Dados carregados com sucesso.")
    print(f"Arquivo........: {DATA_PATH}")
    print(f"N amostras.....: {N}")
    print(f"p features.....: {p}")
    print(f"Classes (C)....: {C} -> {classes.tolist()}")
    print("Faixas (min/max) por sensor:")
    print(f"  Sensor 1: [{X_all[:,0].min():.2f}, {X_all[:,0].max():.2f}]")
    print(f"  Sensor 2: [{X_all[:,1].min():.2f}, {X_all[:,1].max():.2f}]")
    print("==============================================")

    # ------------------------------------------------
    # ETAPA 2 — Visualização inicial
    # ------------------------------------------------
    if SHOW_PLOTS:
        labels_txt = ["Neutro", "Sorriso", "Sobrancelhas Levantadas", "Surpreso", "Rabugento"]
        plt.figure(figsize=(7,6))
        for c in range(1, C+1):
            mask = (y_all == c)
            plt.scatter(X_all[mask, 0], X_all[mask, 1], label=labels_txt[c-1], edgecolors='k', alpha=0.7)
        plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
        plt.ylabel("Sensor 2 (Zigomático Maior)")
        plt.title("Dispersão das classes (EMG)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------
    # ETAPA 3 — Modelos (MQO, Gaussianos, etc.)
    # (vamos implementar quando você pedir)
    # ------------------------------------------------

    # ETAPA 3 — MQO tradicional (one-vs-rest)
    mqo = MQO(add_intercept=True, C=C).fit(X_MQO, y_all)
    print("[MQO] Modelo treinado. Formato da matriz B:", mqo.B.shape)
    # (opcional) teste rápido de re-substituição só para ver se está ok:
    y_pred_resub = mqo.predict(X_MQO)
    acc_resub = np.mean(y_pred_resub == y_all)
    print(f"[MQO] Acurácia de re-substituição (treino no próprio conjunto): {acc_resub:.4f}")

    # ------------------------------------------------
    # ETAPA 3 — Classificador Gaussiano Tradicional (professor)
    # ------------------------------------------------
    # A classe do professor espera:
    #   X_train no formato (p, N)
    #   y_train no formato (1, N)
    # Já temos X_BAYES = X_all.T (p, N). Vamos só ajustar y:
    y_bayes = y_all.reshape(1, -1)          # (1, N)

    gc_trad = GaussTradicional(X_BAYES, y_bayes)
    gc_trad.fit()
    print("[Gaussiano Tradicional] Treinado com sucesso.",
          f"C={gc_trad.C} classes, p={gc_trad.p} features, N={gc_trad.N} amostras.")

    # teste rápido de sanidade (re-substituição):
    #Predizer todas as amostras com loop (a classe do professor é por amostra).
    #Monte Carlo e vetorização vêm depois.
    preds = np.array([gc_trad.predict(X_BAYES[:, [j]]) for j in range(N)], dtype=int).ravel()
    acc_resub = np.mean(preds == y_all)
    print(f"[Gaussiano Tradicional] Acurácia (re-substituição): {acc_resub:.4f}")

    # ------------------------------------------------
    # ETAPA 4 — Validação Monte Carlo (R=500)
    # ------------------------------------------------

    R = 500  # pode testar com R=5 primeiro

    # listas de acurácia por modelo
    acc_MQO = []
    acc_GTRAD = []

    rng = np.random.default_rng(42)  # semente reprodutível

    for r in range(R):
        # 1) split 80/20 (um split por rodada, reaproveitado por TODOS os modelos)
        idx = rng.permutation(N)
        ntr = int(0.8 * N)
        tr_idx, te_idx = idx[:ntr], idx[ntr:]

        X_treino, y_treino = X_all[tr_idx], y_all[tr_idx]
        X_teste, y_teste = X_all[te_idx], y_all[te_idx]

        # 2) --- MQO tradicional (one-vs-rest) ---
        mqo = MQO(add_intercept=True, C=C).fit(X_treino, y_treino)
        yhat_mqo = mqo.predict(X_teste)
        acc_MQO.append(np.mean(yhat_mqo == y_teste))

        # 3) --- Gaussiano Tradicional (Cirillo) ---
        # a classe espera (p,N) e (1,N)
        Xtr_b = X_treino.T
        ytr_b = y_treino.reshape(1, -1)
        Xte_b = X_teste.T

        # gc = GaussianClassifier(Xtr_b, ytr_b)
        # gc.fit()

        # predict é por amostra (coluna): fazemos uma lista-comprehension
        # yhat_gtrad = gc.predict #np.array([gc.predict(Xte_b[:, [j]]) for j in range(Xte_b.shape[1])], dtype=int).ravel()
        # acc_GTRAD.append(np.mean(yhat_gtrad == yte))
        bp = 1
        # acc_GTRAD.append(yhat_gtrad)
        # (opcional) feedback de progresso
        if (r + 1) % 50 == 0:
            print(f"[Monte Carlo] rodadas concluídas: {r+1}/{R}")

    # 4) resumo rápido dessas duas listas (os demais modelos virão depois)
    def resumo_stats(v):
        v = np.array(v, dtype=float)
        return np.mean(v), np.std(v, ddof=1), np.min(v), np.max(v)
    print(np.mean(acc_MQO))
    print("\n================  RESULTADOS PARCIAIS (Acurácia)  ================")
    print(f"{'Modelo':35s}  {'Média':>8s}  {'Desv.Pad.':>10s}  {'Maior':>8s}  {'Menor':>8s}")
    # for nome, accs in [
    #     #("MQO tradicional", acc_MQO),
    #     ("Gaussiano Tradicional", acc_GTRAD),
    # ]:
        # m, s, vmin, vmax = resumo_stats(accs)
        # print(f"{nome:35s}  {m:8.4f}  {s:10.4f}  {vmax:8.4f}  {vmin:8.4f}")


    # ------------------------------------------------
    # ETAPA 5 — Tabelas/Gráficos de resultados
    # (vamos implementar quando você pedir)
    # ------------------------------------------------

if __name__ == "__main__":
    main()
