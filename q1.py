import numpy as np
from numpy.linalg import pinv
from helpers.q1_helper import predicao, plotar_grafico_1, plotar_modelo_regressao, imprimir_tabela_resultados, RSS, exportar_dados_para_csv, poly_design, fit_ridge
import os
import matplotlib.pyplot as plt



# True mostras faz mostrar todos os gráficos
MOSTRAR_GRAFICOS = False

GRAU_POLI = 5         
LAMBDA_POLI = 0.0     # 0 = MQO


# *********************************************************** #
#                           DADOS                             #
# *********************************************************** #

# Carregar os dados
dados = np.loadtxt('dados/aerogerador.dat', delimiter='\t')

# Transpor os dados
dados_t = dados.T

# Gerar a Matriz X e o vetor y
X_sem_intercepor = dados[:, :-1]
y = dados[:, -1:]  

# Plotando o gráfico de dispersão
if (MOSTRAR_GRAFICOS):
    plotar_grafico_1(X_sem_intercepor, y)

plt.figure(figsize=(7,5))
plt.scatter(X_sem_intercepor, y, s=10, edgecolors='k', alpha=0.5)
plt.xlabel("Velocidade do vento"); plt.ylabel("Potência gerada")
plt.title("Dispersão inicial — aerogerador")
plt.tight_layout()
plt.savefig("figsQ1/00_dispersao.png", dpi=200, bbox_inches="tight")
plt.close()

# Pegando os tamanhos de N e y
N, p = X_sem_intercepor.shape

# Adicionando 1s à primera coluna de X (interceptador)
X_com_interceptor = np.c_[np.ones(N), X_sem_intercepor]

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

rss_polinomial_d3 = []


# *********************************************************** #
#                     TREINAMENTO E TESTE                     #
# *********************************************************** #

X_all_full = X_sem_intercepor.reshape(-1, 1)  # (N,1)
y_all_full = y.reshape(-1, 1)                 # (N,1)

for i in range(1000):

    # ***********  Gerando os dados  *********** #

    # Embaralhar os dados
    idx = np.random.permutation(N)

    # Sem intercepto
    X_rodada_sem_interceptor = X_sem_intercepor[idx,:] # X embaralhado
    X_treino_sem_interceptor = X_rodada_sem_interceptor[:int(N*.8),:] # 80% para treino
    # Coloquei aqui, mas não serve de nada
    X_teste_sem_interceptor = X_rodada_sem_interceptor[int(N*.8):,:] # 20% para teste

    # Com intercepto
    X_rodada_com_interceptor = X_com_interceptor[idx,:] # X_interceptor  embaralhado
    X_treino_com_interceptor = X_rodada_com_interceptor[:int(N*.8),:] # 80% para treino
    X_teste_com_interceptor = X_rodada_com_interceptor[int(N*.8):,:] # 20% para teste
    
    # y
    y_rodada = y[idx,:] # y embaralhado
    y_treino = y_rodada[:int(N*.8),:] # 80% para treino
    y_teste = y_rodada[int(N*.8):,:] # 20% para teste

    # ***********  Treinamento dos modelos  *********** #

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



    # --- Polinomial (grau = GRAU_POLI) ---
    from helpers.q1_helper import poly_design, fit_ridge

    Phi_tr = poly_design(X_treino_sem_interceptor, GRAU_POLI)
    Phi_te = poly_design(X_teste_sem_interceptor, GRAU_POLI)

    beta_poly = fit_ridge(Phi_tr, y_treino, lam=LAMBDA_POLI)
    y_pred_poly = Phi_te @ beta_poly
    rss_polinomial_d3.append(RSS(y_teste, y_pred_poly))




    # ***********  Teste de desempenho  *********** #

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

    # ***********  Gráficos  *********** #

    if (MOSTRAR_GRAFICOS):
        plotar_modelo_regressao(X_treino_sem_interceptor, y_treino, beta_hat_MQO_sem_interceptor, "linear", "MQO sem interceptor")
        
        plotar_modelo_regressao(X_treino_sem_interceptor, y_treino, beta_hat_MEDIA[0], "media", "Modelo média")

        for valor_lambda, beta in betas_hat_MQO_regularizado.items():
            nome_modelo = f"MQO regularizado (lambda={valor_lambda})"
            plotar_modelo_regressao(X_treino_sem_interceptor, y_treino, beta, "linear", nome_modelo)

    if MOSTRAR_GRAFICOS and i == 0:
        plotar_modelo_regressao(X_treino_sem_interceptor, y_treino,
                            beta_poly, "polinomial",
                            f"Polinomial (d={GRAU_POLI})", grau=GRAU_POLI)

# *********************************************************** #
#                  CALCULANDO AS ESTATÍSTICAS                 #
# *********************************************************** #

os.makedirs("figsQ1", exist_ok=True)

# ATENÇÃO: corrija os rótulos dos lambdas (0.25, 0.50, 0.75)
labels = [
    "Média da variável dependente",
    "MQO sem intercepto",
    "MQO tradicional",
    "MQO regularizado (0.25)",
    "MQO regularizado (0.50)",
    "MQO regularizado (0.75)",
    "MQO regularizado (1.00)",
    "Regressão Polinomial (d=3)",
]
series = [
    rss_media,
    rss_mqo_sem_interceptor,
    rss_mqo_regularizado_0,     # seu "tradicional" com lam=0
    rss_mqo_regularizado_25,
    rss_mqo_regularizado_50,
    rss_mqo_regularizado_75,
    rss_mqo_regularizado_100,
    rss_polinomial_d3,
]

means = [np.mean(s) for s in series]
stds  = [np.std(s, ddof=1) for s in series]



with open("figsQ1/resumo_rss.csv", "w", encoding="utf-8") as f:
    f.write("modelo,media,desvio_padrao,maior,menor\n")
    for name, s in zip(labels, series):
        m, sd, mx, mn = np.mean(s), np.std(s, ddof=1), np.max(s), np.min(s)
        f.write(f"{name},{m:.6f},{sd:.6f},{mx:.6f},{mn:.6f}\n")
print("CSV salvo em figsQ1/resumo_rss.csv")


plt.figure(figsize=(10,4))
x = np.arange(len(labels))
plt.bar(x, means, yerr=stds, capsize=4)
plt.xticks(x, labels, rotation=35, ha="right")
plt.ylabel("RSS (teste)")
plt.title("RSS médio por modelo (barras = desvio-padrão)")
plt.tight_layout()
plt.savefig("figsQ1/01_bar_media_rss.png", dpi=200, bbox_inches="tight")
plt.close()


plt.figure(figsize=(10,5))
plt.boxplot(series, tick_labels=labels, showmeans=True)
plt.ylabel("RSS (teste)")
plt.title("Distribuição de RSS — Monte Carlo")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("figsQ1/02_boxplot_rss.png", dpi=200, bbox_inches="tight")
plt.close()


plt.figure(figsize=(8,5))
for name, s in zip(labels, series):
    s_sorted = np.sort(np.array(s))
    y = np.linspace(0, 1, len(s_sorted))
    plt.plot(s_sorted, y, label=name, alpha=0.9)
plt.xlabel("RSS (teste)"); plt.ylabel("Fração acumulada")
plt.title("CDF das RSS por modelo")
plt.legend(fontsize=8)
plt.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig("figsQ1/03_cdf_rss.png", dpi=200, bbox_inches="tight")
plt.close()



Phi_all = poly_design(X_all_full, GRAU_POLI)
beta_poly_all = fit_ridge(Phi_all, y_all_full, lam=LAMBDA_POLI)

xgrid = np.linspace(X_all_full.min(), X_all_full.max(), 400).reshape(-1,1)
Phi_grid = poly_design(xgrid, GRAU_POLI)
ygrid = Phi_grid @ beta_poly_all

plt.figure(figsize=(7,5))
plt.scatter(X_all_full, y_all_full, s=8, edgecolors='k', alpha=0.35, label="Observações")
plt.plot(xgrid, ygrid, lw=2, label=f"Polinomial (d={GRAU_POLI})")
plt.xlabel("Velocidade do vento"); plt.ylabel("Potência do aerogerador")
plt.title("Ajuste polinomial ilustrativo (sobre todo o conjunto)")
plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
plt.savefig("figsQ1/04_curva_polinomial.png", dpi=200, bbox_inches="tight")
plt.close()


# Escolha 1 split 80/20
idx = np.random.permutation(len(X_all_full))
ntr = int(0.8*len(X_all_full))
Xtr, ytr = X_all_full[idx[:ntr]], y_all_full[idx[:ntr]]
Xte, yte = X_all_full[idx[ntr:]], y_all_full[idx[ntr:]]

# MQO tradicional (linear) nesse split
Xtr_lin = np.hstack([np.ones((Xtr.shape[0],1)), Xtr])
Xte_lin = np.hstack([np.ones((Xte.shape[0],1)), Xte])
beta_lin = np.linalg.pinv(Xtr_lin.T @ Xtr_lin) @ (Xtr_lin.T @ ytr)
res_lin = yte - (Xte_lin @ beta_lin)

# Polinomial (d=3) nesse split
Phi_tr = poly_design(Xtr, GRAU_POLI)
Phi_te = poly_design(Xte, GRAU_POLI)
beta_poly = fit_ridge(Phi_tr, ytr, lam=LAMBDA_POLI)
res_poly = yte - (Phi_te @ beta_poly)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.hist(res_lin, bins=25, edgecolor='k')
plt.title("Resíduos — Linear"); plt.xlabel("erro"); plt.ylabel("freq")
plt.subplot(1,2,2); plt.hist(res_poly, bins=25, edgecolor='k')
plt.title("Resíduos — Polinomial d=3"); plt.xlabel("erro")
plt.tight_layout()
plt.savefig("figsQ1/05_hist_residuos_linear_vs_poly.png", dpi=200, bbox_inches="tight")
plt.close()



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
    "Regressão Polinomial (d=3)": {
    "media": np.mean(rss_polinomial_d3),
    "desvio_padrao": np.std(rss_polinomial_d3),
    "maior_valor": np.max(rss_polinomial_d3),
    "menor_valor": np.min(rss_polinomial_d3),
},
}

# *********************************************************** #
#                    PRINTANDO OS RESULTADOS                  #
# *********************************************************** #

imprimir_tabela_resultados(resultados_para_tabela)
exportar_dados_para_csv(resultados_para_tabela)

# =================== FIGURAS: Linhas ajustadas (modelos lineares) ===================
# Pré-requisitos já definidos no script:
# - X_sem_intercepor: (N,1) com a velocidade
# - y: (N,1) com a potência
# - X_all_full, y_all_full: aliases (N,1) preservados (se ainda não criou, faça:)
# X_all_full = X_sem_intercepor.reshape(-1,1)
# y_all_full = y.reshape(-1,1)

# 1) Design de grau 1 (equivale a linear com intercepto): Φ = [1, x]
Phi_lin = poly_design(X_all_full, d=1)  # usa helper que você já criou

# 2) MQO tradicional (lam=0) e Ridge com vários lambdas
beta_lin_mqo = fit_ridge(Phi_lin, y_all_full, lam=0.0)
beta_lin_l25 = fit_ridge(Phi_lin, y_all_full, lam=0.25)
beta_lin_l50 = fit_ridge(Phi_lin, y_all_full, lam=0.50)
beta_lin_l75 = fit_ridge(Phi_lin, y_all_full, lam=0.75)
beta_lin_l100 = fit_ridge(Phi_lin, y_all_full, lam=1.00)

# 3) MQO sem intercepto (reta forçada na origem)
#    Fórmula fechada: slope = (x^T y)/(x^T x)
num = float((X_all_full.T @ y_all_full)[0,0])
den = float((X_all_full.T @ X_all_full)[0,0]) + 1e-12
slope_no_intercept = num / den
beta_no_intercept = np.array([[0.0], [slope_no_intercept]])  # [β0=0; β1]

# 4) Grid para plotar as retas
xgrid = np.linspace(X_all_full.min(), X_all_full.max(), 400).reshape(-1,1)
Phi_grid_lin = poly_design(xgrid, d=1)

# 5) Previsões das retas
y_mqo   = Phi_grid_lin @ beta_lin_mqo
y_l25   = Phi_grid_lin @ beta_lin_l25
y_l50   = Phi_grid_lin @ beta_lin_l50
y_l75   = Phi_grid_lin @ beta_lin_l75
y_l100  = Phi_grid_lin @ beta_lin_l100
y_noint = beta_no_intercept[0,0] + beta_no_intercept[1,0] * xgrid

# 6) Plot (dispersão + todas as retas lineares)
plt.figure(figsize=(8,5))
plt.scatter(X_all_full, y_all_full, s=8, edgecolors='k', alpha=0.35, label="Observações")
plt.plot(xgrid, y_noint,  linewidth=2, label="MQO sem intercepto")
plt.plot(xgrid, y_mqo,    linewidth=2, label="MQO tradicional")
plt.plot(xgrid, y_l25,    linewidth=1.8, label="Tikhonov λ=0.25")
plt.plot(xgrid, y_l50,    linewidth=1.8, label="Tikhonov λ=0.50")
plt.plot(xgrid, y_l75,    linewidth=1.8, label="Tikhonov λ=0.75")
plt.plot(xgrid, y_l100,   linewidth=1.8, label="Tikhonov λ=1.00")
plt.xlabel("Velocidade do vento"); plt.ylabel("Potência gerada")
plt.title("Ajustes lineares (sem intercepto, MQO e Ridge)")
plt.legend(fontsize=8); plt.grid(True, alpha=.3); plt.tight_layout()
plt.savefig("figsQ1/06_linhas_lineares.png", dpi=200, bbox_inches="tight")
plt.close()


Phi_all_poly = poly_design(X_all_full, GRAU_POLI)
beta_poly_all = fit_ridge(Phi_all_poly, y_all_full, lam=0.0)
Phi_grid_poly = poly_design(xgrid, GRAU_POLI)
y_poly = Phi_grid_poly @ beta_poly_all

plt.figure(figsize=(8,5))
plt.scatter(X_all_full, y_all_full, s=8, edgecolors='k', alpha=0.35, label="Observações")
plt.plot(xgrid, y_mqo,  linewidth=2, label="MQO (linear)")
plt.plot(xgrid, y_poly, linewidth=2, label=f"Polinomial (d={GRAU_POLI})")
plt.xlabel("Velocidade do vento"); plt.ylabel("Potência gerada")
plt.title("Comparação: Linear vs Polinomial")
plt.legend(fontsize=9); plt.grid(True, alpha=.3); plt.tight_layout()
plt.savefig("figsQ1/07_linear_vs_polinomial.png", dpi=200, bbox_inches="tight")
plt.close()


# ================ Histograma de resíduos (ilustrativo) ================
# Resíduos no conjunto inteiro (apenas para comparar formas)
res_lin  = y_all_full - (Phi_lin @ beta_lin_mqo)
res_poly = y_all_full - (Phi_all_poly @ beta_poly_all)

plt.figure(figsize=(9,4))
plt.subplot(1,2,1); plt.hist(res_lin,  bins=30, edgecolor='k');  plt.title("Resíduos — Linear (MQO)")
plt.xlabel("erro"); plt.ylabel("freq")
plt.subplot(1,2,2); plt.hist(res_poly, bins=30, edgecolor='k');  plt.title(f"Resíduos — Polinomial (d={GRAU_POLI})")
plt.xlabel("erro")
plt.tight_layout()
plt.savefig("figsQ1/08_hist_residuos_linear_vs_poly.png", dpi=200, bbox_inches="tight")
plt.close()
