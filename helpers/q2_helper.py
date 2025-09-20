import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

def grafico_dispersao_inicial(X, y, save_path=None, show=True):
    nomes = ["Neutro", "Sorriso", "Sobrancelhas Levantadas", "Surpreso", "Rabugento"]
    cores = ["red", "blue", "green", "yellow", "cyan"]

    plt.figure()
    for c in range(1, 6):
        m = (y == c)
        plt.scatter(X[m, 0], X[m, 1], c=cores[c-1], label=nomes[c-1], edgecolors='k', s=20)
    plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
    plt.ylabel("Sensor 2 (Zigomático Maior)")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def calculate_accuracy(y_teste, y_pred):
    return np.mean(y_teste == y_pred)

# Tentativa do Gemini. Não gostei
def plotar_regioes_decisao_dinamico(model, X, y, title, save_path=None, show=True):
    unique_classes = np.unique(y)
    cores_regioes = ['black','orange','purple','pink','brown']
    marcadores = ['o','o','o','o','o']

    x_min, x_max = X[:, 0].min() - 20, X[:, 0].max() + 20
    y_min, y_max = X[:, 1].min() - 20, X[:, 1].max() + 20
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 20),
                         np.arange(y_min, y_max, 20))

    a = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(a).reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    # 1) regiões primeiro
    plt.contourf(xx, yy, Z, alpha=0.7, cmap=ListedColormap(cores_regioes))
    # 2) pontos por cima
    for i, classe in enumerate(unique_classes):
        plt.scatter(X[y == classe, 0], X[y == classe, 1],
                    c=cores_regioes[i], label=f'Classe {classe}',
                    marker=marcadores[i], edgecolors='k', s=80)
    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()



def resumo_stats(v):
    v = np.array(v, dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan, np.nan  # caso a lista não tenha sido preenchida
    return np.mean(v), np.std(v, ddof=1), np.min(v), np.max(v)  # dp amostral




#Plots para relatorio:


# --- helper: grade + fronteira ---
def plot_decision_boundary(predict_fn, X, y, title,
                           save_path=None, show=True,
                           class_names=None, colors=None):
    classes = np.unique(y)

    # nomes default (se não passar nada)
    if class_names is None:
        class_names = ["Neutro", "Sorriso", "Sobrancelhas Levantadas", "Surpreso", "Rabugento"]

    # mapeia rótulo inteiro -> nome amigável
    name_map = {c: (class_names[i] if i < len(class_names) else f"Classe {c}")
                for i, c in enumerate(classes)}
    
    x_min, x_max = X[:,0].min()-50, X[:,0].max()+50
    y_min, y_max = X[:,1].min()-50, X[:,1].max()+50
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]          # (Ngrid, 2)
    y_grid = predict_fn(grid).reshape(xx.shape)   # (300,300)
    plt.figure(figsize=(6.2,5.2))
    plt.contourf(xx, yy, y_grid, levels=np.arange(1,7)-0.5, alpha=0.25)
    for i, c in enumerate(classes):
        m = (y == c)
        plt.scatter(X[m, 0], X[m, 1],
                    # c=..., (deixe sua cor como já está)
                    label=name_map[c],
                    edgecolors='k', s=10 if X.shape[0] > 1000 else 40, alpha=0.8)
    plt.title(title); plt.xlabel("Sensor 1"); plt.ylabel("Sensor 2")
    plt.legend(loc='upper right', fontsize=8); plt.grid(True, alpha=.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_gaussian_ellipses(means, covs, ax=None, n_std=2.0, save_path=None, show=True, **kw):
    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2,5.2))
    if ax is None: fig, ax = plt.subplots(figsize=(6.2,5.2))
    for i in range(means.shape[0]):
        mu = means[i]; S = covs[i]
        vals, vecs = np.linalg.eigh(S)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:,order]
        theta = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        width, height = 2*n_std*np.sqrt(vals)
        from matplotlib.patches import Ellipse
        e = Ellipse(xy=mu, width=width, height=height, angle=theta, fill=False, **kw)
        ax.add_patch(e)

    if created_ax:
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    return ax


def confusion_matrix_5(y_true, y_pred, C=5):
    M = np.zeros((C,C), dtype=int)
    for t,p in zip(y_true, y_pred):
        M[t-1, p-1] += 1
    return M

def plot_confusion(M, title, save_path=None, show=True):
    plt.figure(figsize=(5.6,4.8))
    plt.imshow(M, interpolation='nearest', cmap='Blues')
    plt.title(title); plt.colorbar()
    ticks = np.arange(5); lbls = ["Neutro","Sorriso","Sobr.","Surpr.","Rabug."]
    plt.xticks(ticks, lbls, rotation=45, ha='right'); plt.yticks(ticks, lbls)
    plt.ylabel('Verdadeiro'); plt.xlabel('Previsto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, M[i,j], ha='center', va='center', color='k', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
