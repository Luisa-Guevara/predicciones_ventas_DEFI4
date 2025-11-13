"""
Utilidades para análisis funcional de ventas y clustering
"""
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from skfda.representation.basis import BSplineBasis
from skfda.representation.grid import FDataGrid


# ===== PALETA DE COLORES FIJA =====
CLUSTER_COLORS = {
    1: "#FF6B35",  # Naranja
    2: "#004E89",  # Azul
    3: "#2A9D8F",  # Verde azulado
    4: "#E76F51",  # Coral
    5: "#8338EC",  # Púrpura
    6: "#F4A261",  # Dorado
}


def load_functional_sales():
    """
    Carga y procesa datos de ventas funcionales
    
    Returns:
        tuple: (df_func, periodos, ventas) o (None, None, None) si hay error
    """
    try:
        df_func = pd.read_csv("data/ventas_funcionales.csv")
        periodos = df_func["Periodo"].values
        ventas = df_func.drop(columns=["Periodo"]).values.T
        return df_func, periodos, ventas
    except Exception as e:
        print(f"Error cargando ventas funcionales: {e}")
        return None, None, None


def find_optimal_k(coef, Z, max_k=6):
    """
    Encuentra el número óptimo de clusters usando Silhouette Score
    
    Args:
        coef: Matriz de coeficientes de B-splines
        Z: Matriz de linkage jerárquico
        max_k: Número máximo de clusters a evaluar
    
    Returns:
        tuple: (k_opt, sil_scores) - número óptimo y diccionario de scores
    """
    sil_scores = {}
    for k in range(2, min(max_k + 1, len(coef))):
        labels_k = fcluster(Z, k, criterion="maxclust")
        sil_scores[k] = silhouette_score(coef, labels_k, metric="euclidean")
    
    k_opt = max(sil_scores, key=sil_scores.get)
    return k_opt, sil_scores


def compute_clusters_automatic(ventas, periodos):
    """
    Calcula clustering jerárquico con K óptimo automático
    
    Args:
        ventas: Matriz de ventas (n_tiendas x n_periodos)
        periodos: Array de periodos temporales
    
    Returns:
        tuple: (labels, fd_eval, eval_points, Z, coef, k_opt, sil_scores)
    """
    n_tiendas, n_periodos = ventas.shape
    eval_points = np.arange(1, n_periodos + 1)
    
    # Suavizado con B-splines
    basis = BSplineBasis(domain_range=(1, n_periodos), n_basis=8)
    fd_grid = FDataGrid(data_matrix=ventas, grid_points=periodos)
    fd_basis = fd_grid.to_basis(basis)
    
    # Clustering en espacio de coeficientes
    coef = fd_basis.coefficients
    dist_matrix = pdist(coef, metric="euclidean")
    Z = linkage(dist_matrix, method="ward")
    
    # Encontrar K óptimo
    k_opt, sil_scores = find_optimal_k(coef, Z)
    labels = fcluster(Z, k_opt, criterion="maxclust")
    
    # Evaluar curvas suavizadas
    fd_eval = fd_basis(eval_points)
    
    return labels, fd_eval, eval_points, Z, coef, k_opt, sil_scores


def assign_cluster_by_sales(prediction, fd_eval, labels, k_clusters):
    """
    Asigna automáticamente el cluster más similar según ventas del mes 24
    
    Args:
        prediction: Valor de ventas predicho para el mes 24
        fd_eval: Matriz de curvas funcionales evaluadas
        labels: Array de etiquetas de cluster
        k_clusters: Número total de clusters
    
    Returns:
        int: ID del cluster asignado
    """
    cluster_means_m24 = []
    
    for k in range(1, k_clusters + 1):
        idx_k = (labels == k)
        curvas_k = fd_eval[idx_k]
        mean_k24 = float(curvas_k[:, -1].mean())
        cluster_means_m24.append((k, mean_k24))
    
    # Encontrar cluster con media más cercana a la predicción
    cluster_asignado = min(cluster_means_m24, key=lambda x: abs(x[1] - prediction))[0]
    return cluster_asignado


def estimate_sales_curve(fd_eval, labels, cluster_id, ventas_mes24):
    """
    Estima curva de ventas ajustada al valor del mes 24
    
    Args:
        fd_eval: Matriz de curvas funcionales evaluadas
        labels: Array de etiquetas de cluster
        cluster_id: ID del cluster objetivo
        ventas_mes24: Valor de ventas conocido para el mes 24
    
    Returns:
        tuple: (estimacion, mean_cluster, curvas_cluster)
    """
    idx_cluster = (labels == cluster_id)
    curvas_cluster = fd_eval[idx_cluster]
    mean_cluster = curvas_cluster.mean(axis=0)
    
    # Ajustar al valor conocido en mes 24
    factor_escalado = ventas_mes24 / float(mean_cluster[-1])
    estimacion = mean_cluster * factor_escalado
    
    return estimacion, mean_cluster, curvas_cluster