import numpy as np
import matplotlib.pyplot as plt
from helpers.q2_helper import (
    calculate_accuracy, grafico_dispersao_inicial, resumo_stats,
    plot_decision_boundary, confusion_matrix_5, plot_confusion
)
import os
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

os.makedirs("figs", exist_ok=True)

# Dispersão inicial (salva em arquivo, sem abrir janela)
grafico_dispersao_inicial(X, y, save_path="figs/01_dispersao.png", show=False)



# Inicializar listas para armazenar as acurácias
acc_MQO = [] # Mínimos Quadrados Ordinários
acc_G_T = [] # Classificador Gaussiano Tradicional
acc_G_C_G = [] # Classificador Gaussiano com Covariâncias Global
acc_G_M_A= [] # Classificador Gaussiano com Matriz Agregada
acuracias_G_M_R = [] # Classificador Gaussiano com Matriz Regularizado
acc_G_B_N = [] # Classificador Gaussiano de Bayes Naive
acc_G_FRIED_025 = []
acc_G_FRIED_050 = []
acc_G_FRIED_075 = []
# *********************************************************** #
#                     TREINAMENTO E TESTE                     #
# *********************************************************** #

# --- Split de demonstração só para figuras ---
idx_demo = np.random.permutation(N)
ntr_demo = int(0.8 * N)
Xtr_demo, ytr_demo = X[idx_demo[:ntr_demo]], y[idx_demo[:ntr_demo]]
Xte_demo, yte_demo = X[idx_demo[ntr_demo:]], y[idx_demo[ntr_demo:]]

# Médias e covariâncias por classe no split demo
means = []; covs = []
for c in np.unique(ytr_demo):
    Xc = Xtr_demo[ytr_demo == c]
    means.append(Xc.mean(axis=0))
    covs.append(np.cov(Xc.T, bias=False))
means = np.array(means); covs = np.array(covs)

from helpers.q2_helper import plot_gaussian_ellipses
plot_gaussian_ellipses(means, covs, Xtr_demo, ytr_demo,
                       "Elipses de Covariância (treino, split demo)",
                       save_path="figs/21_elipses_cov.png", show=False)


# --- MQO (demo) ---

# Treina MQO neste split e plota fronteira
mqo_demo = MQO(add_intercept=True, C=C); mqo_demo.fit(Xtr_demo, ytr_demo)
plot_decision_boundary(lambda G: mqo_demo.predict(G), X, y,
                       "Fronteira — MQO (split demo)",
                       save_path="figs/02_mqo_boundary.png", show=False)

# Matriz de confusão do MQO neste split
yhat_demo_mqo = mqo_demo.predict(Xte_demo)
M_mqo = confusion_matrix_5(yte_demo, yhat_demo_mqo, C=5)
plot_confusion(M_mqo, "Confusão — MQO (split demo)",
               save_path="figs/03_mqo_confusion.png", show=False)

# --- Gauss Tradicional (demo) ---

gc_demo = GaussTradicional(Xtr_demo.T, ytr_demo.reshape(1,-1)); gc_demo.fit()
plot_decision_boundary(lambda G: gc_demo.predict_batch(G.T), X, y,
                       "Fronteira — Gauss Tradicional (demo)",
                       save_path="figs/04_gauss_trad_boundary.png", show=False)
yhat_demo_gc = gc_demo.predict_batch(Xte_demo.T)
M_gc = confusion_matrix_5(yte_demo, yhat_demo_gc, C=5)
plot_confusion(M_gc, "Confusão — Gauss Tradicional (demo)",
               save_path="figs/05_gauss_trad_confusion.png", show=False)

# --- Gauss Covariância Global (demo) ---

gc_global_demo = GaussCovarianciasGlobal(Xtr_demo.T, ytr_demo.reshape(1, -1))
gc_global_demo.fit()

plot_decision_boundary(
    lambda G: gc_global_demo.predict_batch(G.T),  # se não tiver predict_batch, ver OBS no fim
    X, y,
    "Fronteira — Gauss Cov. Global (demo)",
    save_path="figs/06_gauss_global_boundary.png", show=False
)

yhat_demo_global = gc_global_demo.predict_batch(Xte_demo.T)
M_global = confusion_matrix_5(yte_demo, yhat_demo_global, C=5)
plot_confusion(M_global, "Confusão — Gauss Cov. Global (demo)",
               save_path="figs/07_gauss_global_confusion.png", show=False)


# --- Gauss Matriz Agregada / Pooled (demo) ---
gc_pooled_demo = GaussCovarianciaAgregada(Xtr_demo.T, ytr_demo.reshape(1, -1))
gc_pooled_demo.fit()

plot_decision_boundary(
    lambda G: gc_pooled_demo.predict_batch(G.T),
    X, y,
    "Fronteira — Gauss Pooled (demo)",
    save_path="figs/08_gauss_pooled_boundary.png", show=False
)

yhat_demo_pooled = gc_pooled_demo.predict_batch(Xte_demo.T)
M_pooled = confusion_matrix_5(yte_demo, yhat_demo_pooled, C=5)
plot_confusion(M_pooled, "Confusão — Gauss Pooled (demo)",
               save_path="figs/09_gauss_pooled_confusion.png", show=False)


# --- Gauss Naive (demo) ---
gc_naive_demo = GaussNaive(Xtr_demo.T, ytr_demo.reshape(1, -1))
gc_naive_demo.fit()

plot_decision_boundary(
    lambda G: gc_naive_demo.predict_batch(G.T),
    X, y,
    "Fronteira — Gauss Naive (demo)",
    save_path="figs/10_gauss_naive_boundary.png", show=False
)

yhat_demo_naive = gc_naive_demo.predict_batch(Xte_demo.T)
M_naive = confusion_matrix_5(yte_demo, yhat_demo_naive, C=5)
plot_confusion(M_naive, "Confusão — Gauss Naive (demo)",
               save_path="figs/11_gauss_naive_confusion.png", show=False)


# --- Gauss Friedman λ=0.25 (demo) ---
gc_l025_demo = GaussTradicional(Xtr_demo.T, ytr_demo.reshape(1, -1), lam=0.25)
gc_l025_demo.fit()
plot_decision_boundary(
    lambda G: gc_l025_demo.predict_batch(G.T),
    X, y,
    "Fronteira — Gauss Friedman λ=0.25 (demo)",
    save_path="figs/12_friedman_025_boundary.png", show=False
)
yhat_demo_l025 = gc_l025_demo.predict_batch(Xte_demo.T)
M_l025 = confusion_matrix_5(yte_demo, yhat_demo_l025, C=5)
plot_confusion(M_l025, "Confusão — Gauss Friedman λ=0.25 (demo)",
               save_path="figs/13_friedman_025_confusion.png", show=False)

# --- Gauss Friedman λ=0.50 (demo) ---
gc_l050_demo = GaussTradicional(Xtr_demo.T, ytr_demo.reshape(1, -1), lam=0.50)
gc_l050_demo.fit()
plot_decision_boundary(
    lambda G: gc_l050_demo.predict_batch(G.T),
    X, y,
    "Fronteira — Gauss Friedman λ=0.50 (demo)",
    save_path="figs/14_friedman_050_boundary.png", show=False
)
yhat_demo_l050 = gc_l050_demo.predict_batch(Xte_demo.T)
M_l050 = confusion_matrix_5(yte_demo, yhat_demo_l050, C=5)
plot_confusion(M_l050, "Confusão — Gauss Friedman λ=0.50 (demo)",
               save_path="figs/15_friedman_050_confusion.png", show=False)

# --- Gauss Friedman λ=0.75 (demo) ---
gc_l075_demo = GaussTradicional(Xtr_demo.T, ytr_demo.reshape(1, -1), lam=0.75)
gc_l075_demo.fit()
plot_decision_boundary(
    lambda G: gc_l075_demo.predict_batch(G.T),
    X, y,
    "Fronteira — Gauss Friedman λ=0.75 (demo)",
    save_path="figs/16_friedman_075_boundary.png", show=False
)
yhat_demo_l075 = gc_l075_demo.predict_batch(Xte_demo.T)
M_l075 = confusion_matrix_5(yte_demo, yhat_demo_l075, C=5)
plot_confusion(M_l075, "Confusão — Gauss Friedman λ=0.75 (demo)",
               save_path="figs/17_friedman_075_confusion.png", show=False)
# *********************************************************** #

contador_progresso = 0
interacoes = 5
for count in range(interacoes):
    # ***********  Aleatorizando os dados  *********** #
    
    idx = np.random.permutation(N)
    X_rodada = X_mqo_features[idx,:] # X embaralhado
    Y_rodada = Y_mqo[idx] # Y embaralhado

    X_treino = X_rodada[:int(N*.8),:] # 80% para treino
    y_treino = Y_rodada[:int(N*.8)] # 80% para treino

    X_teste = X_rodada[int(N*.8):,:] # 20% para teste
    y_teste = Y_rodada[int(N*.8):] # 20% para teste

    #X_teste_sample = X_rodada[int(0.8 * X.shape[0]), :].reshape(1,2) # Prof
    
    # ***********  Instanciar e treinar os modelos  *********** #

    modelo_MQO = MQO(add_intercept = True, C = C)
    modelo_G_T = GaussTradicional(X_treino.T, y_treino.T.reshape(1,len(y_treino)))
    modelo_G_C_G = GaussCovarianciasGlobal(X_treino.T, y_treino.T.reshape(1,len(y_treino)))
    modelo_G_B_N = GaussNaive(X_treino.T, y_treino.T.reshape(1,len(y_treino)))
    modelo_G_M_A = GaussCovarianciaAgregada(X_treino.T, y_treino.T.reshape(1,len(y_treino)))

    gc_l025 = GaussTradicional(X_treino.T, y_treino.T.reshape(1,len(y_treino)), lam=0.25)
    gc_l050 = GaussTradicional(X_treino.T, y_treino.T.reshape(1,len(y_treino)), lam=0.50)
    gc_l075 = GaussTradicional(X_treino.T, y_treino.T.reshape(1,len(y_treino)), lam=0.75)


    modelo_MQO.fit(X_treino, y_treino)
    modelo_G_T.fit()
    modelo_G_C_G.fit()
    modelo_G_B_N.fit()
    modelo_G_M_A.fit()
    gc_l025.fit()
    gc_l050.fit()
    gc_l075.fit()

    # ***********  Predições  *********** #

    #Antiga Errada:
    # predicao_MQO = modelo_MQO.predict(X_teste)
    # predicao_G_T = modelo_G_T.predict(X_teste_sample)
    # predicao_G_C_G = modelo_G_C_G.predict(X_teste_sample)
    # predicao_G_B_N = modelo_G_B_N.predict(X_teste_sample)
    # predicao_G_M_A = modelo_G_M_A.predict(X_teste_sample)

    #Antiga Lenta:
    # predicao_MQO = modelo_MQO.predict(X_teste)

    # # Gauss Tradicional (toda a base de teste)
    # preds_G_T = np.array([modelo_G_T.predict(X_teste[j, :].reshape(1,2).T)
    #                     for j in range(X_teste.shape[0])], dtype=int)

    # # Gauss Covariâncias Global
    # preds_G_C_G = np.array([modelo_G_C_G.predict(X_teste[j, :].reshape(1,2).T)
    #                         for j in range(X_teste.shape[0])], dtype=int)

    # # Gauss Naive
    # preds_G_B_N = np.array([modelo_G_B_N.predict(X_teste[j, :].reshape(1,2).T)
    #                         for j in range(X_teste.shape[0])], dtype=int)

    # # Gauss Matriz Agregada (pooled)
    # preds_G_M_A = np.array([modelo_G_M_A.predict(X_teste[j, :].reshape(1,2).T)
    #                         for j in range(X_teste.shape[0])], dtype=int)
    
    # # Gauss Friedman λ=0.25
    # yhat_l025 = np.array([gc_l025.predict(X_teste[j, :].reshape(1,2).T)
    #                         for j in range(X_teste.shape[0])], dtype=int)
    
    # yhat_l050 = np.array([gc_l050.predict(X_teste[j, :].reshape(1,2).T)
    #                         for j in range(X_teste.shape[0])], dtype=int)
    
    # yhat_l075 = np.array([gc_l075.predict(X_teste[j, :].reshape(1,2).T)
    #                         for j in range(X_teste.shape[0])], dtype=int)

    #Nova Rápida:

    # MQO já é batch
    predicao_MQO = modelo_MQO.predict(X_teste)

    # Para todos os Gaussianos: use batch no formato (p, N_te)
    Xte_T = X_teste.T

    # Gauss Tradicional
    preds_G_T   = modelo_G_T.predict_batch(Xte_T)

    # Gauss Covariâncias Global (uma Σ global)
    preds_G_C_G = modelo_G_C_G.predict_batch(Xte_T)

    # Gauss Naive (diagonal)
    preds_G_B_N = modelo_G_B_N.predict_batch(Xte_T)

    # Gauss Matriz Agregada / Pooled (uma Σ pooled)
    preds_G_M_A = modelo_G_M_A.predict_batch(Xte_T)

    # Gauss Friedman (λ = 0.25, 0.50, 0.75)
    yhat_l025   = gc_l025.predict_batch(Xte_T)
    yhat_l050   = gc_l050.predict_batch(Xte_T)
    yhat_l075   = gc_l075.predict_batch(Xte_T)

    # ***********  Acurácias  *********** #
    
    # acc_MQO.append(np.mean(predicao_MQO == y_teste))
    # acc_G_T.append(calculate_accuracy(y_teste.T, predicao_G_T))
    # acc_G_C_G.append(calculate_accuracy(y_teste.T, predicao_G_C_G))
    # acc_G_B_N.append(calculate_accuracy(y_teste.T, predicao_G_B_N))
    # acc_G_M_A.append(calculate_accuracy(y_teste.T, predicao_G_M_A))

    acc_MQO.append(np.mean(predicao_MQO == y_teste))
    acc_G_T.append(calculate_accuracy(y_teste, preds_G_T))
    acc_G_C_G.append(calculate_accuracy(y_teste, preds_G_C_G))
    acc_G_B_N.append(calculate_accuracy(y_teste, preds_G_B_N))
    acc_G_M_A.append(calculate_accuracy(y_teste, preds_G_M_A))
    acc_G_FRIED_025.append(calculate_accuracy(y_teste, yhat_l025))
    acc_G_FRIED_050.append(calculate_accuracy(y_teste, yhat_l050))
    acc_G_FRIED_075.append(calculate_accuracy(y_teste, yhat_l075))

    
    # ***********  Contador de progresso  *********** #

    if(contador_progresso  % 50 == 0):
        print(f"{contador_progresso}/500")
    contador_progresso = contador_progresso + 1

print(f"{contador_progresso}/500\nFinalizado\n")

# *********************************************************** #
#                            PRINTS                           #
# *********************************************************** #

media_MQO = np.mean(acc_MQO)
media_G_T = np.mean(acc_G_T)
media_G_C_G = np.mean(acc_G_C_G)
media_G_B_N = np.mean(acc_G_B_N)
media_G_M_A = np.mean(acc_G_M_A)
media_G_FRIED_025 = np.mean(acc_G_FRIED_025)
media_G_FRIED_050 = np.mean(acc_G_FRIED_050)
media_G_FRIED_075 = np.mean(acc_G_FRIED_075)


# print(f"MQO: {media_MQO:.4f}")
# print(f"Gauss Tradicional: {media_G_T}")
# print(f"Gauss Covariância Global: {media_G_C_G}")
# print(f"Gauss Naive: {media_G_B_N}")
# print(f"Gauss Matriz Agregada: {media_G_M_A}")
# print(f"Gauss Friedman λ=0.25: {media_G_FRIED_025}")
# print(f"Gauss Friedman λ=0.50: {media_G_FRIED_050}")
# print(f"Gauss Friedman λ=0.75: {media_G_FRIED_075}")
# # *********************************************************** #


print("\n================  RESULTADOS (Acurácia em TESTE)  ================")
print(f"{'Modelo':35s}  {'Média':>8s}  {'Desv.Pad.':>10s}  {'Maior':>8s}  {'Menor':>8s}")

tabela = [
    ("MQO",                         acc_MQO),
    ("Gauss Tradicional",           acc_G_T),
    ("Gauss Cov. Global",           acc_G_C_G),
    ("Gauss Naive",                 acc_G_B_N),
    ("Gauss Matriz Agregada",       acc_G_M_A),
    ("Gauss Friedman λ=0.25",       acc_G_FRIED_025),
    ("Gauss Friedman λ=0.50",       acc_G_FRIED_050),
    ("Gauss Friedman λ=0.75",       acc_G_FRIED_075),
]

nomesAbreviados = [
    ("MQO"),
    ("Gauss Trad."),
    ("Gauss Cov. Global"),
    ("Gauss Naive"),
    ("Gauss M. Agreg."),
    ("G. Fried. λ=0.25"),
    ("G. Fried. λ=0.50"),
    ("G. Fried. λ=0.75"),
]

for nome, accs in tabela:
    m, s, vmax, vmin = resumo_stats(accs)
    print(f"{nome:35s}  {m:8.4f}  {s:10.4f}  {vmax:8.4f}  {vmin:8.4f}")

plt.figure(figsize=(12,5))
labels = nomesAbreviados
dados  = [x[1] for x in tabela]
plt.boxplot(dados, tick_labels=labels, showmeans=True)
plt.ylabel("Acurácia (teste)")
plt.title("Distribuição das Acurácias — Monte Carlo")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figs/18_boxplot_acuracias.png", dpi=200, bbox_inches="tight")
plt.close()


# --- Efeito do λ na acurácia (Gauss Friedman) ---
lambdas = np.array([0.25, 0.50, 0.75])
means   = np.array([np.mean(acc_G_FRIED_025), np.mean(acc_G_FRIED_050), np.mean(acc_G_FRIED_075)])
stds    = np.array([np.std(acc_G_FRIED_025, ddof=1), np.std(acc_G_FRIED_050, ddof=1), np.std(acc_G_FRIED_075, ddof=1)])

plt.figure(figsize=(6,4))
plt.errorbar(lambdas, means, yerr=stds, fmt='-o', capsize=4)
plt.xlabel("λ (Friedman)"); plt.ylabel("Acurácia (teste)")
plt.title("Efeito de λ no Gauss Regularizado")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("figs/19_lambda_vs_acc.png", dpi=200, bbox_inches="tight")
plt.close()


# Normaliza por linha (percentual por classe verdadeira)
M_gc_pct = M_gc.astype(float) / M_gc.sum(axis=1, keepdims=True)
# Reusa o plot existente; ele aceita floats (vai mostrar decimais)
plot_confusion(M_gc_pct, "Confusão Normalizada — Gauss Tradicional (demo)",
               save_path="figs/20_confusao_gauss_trad_pct.png", show=False)


# Gráfico de dispersão dos dados
# grafico_dispersao_inicial(dados, classe_ids)



