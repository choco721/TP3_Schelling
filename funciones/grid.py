"""
grid.py — Inicialización y utilidades de grilla para el modelo de Schelling.
Grilla 2D con condiciones de borde periódicas (toroidal).
Valores: 0 = vacío, 1 = Grupo A (naranja), 2 = Grupo B (verde)
"""

import numpy as np


def inicializar_grilla(N: int, frac_vacia: float = 0.20, seed: int = 42) -> np.ndarray:
    """
    Crea una grilla N×N con dos grupos distribuidos aleatoriamente.
    
    Parámetros
    ----------
    N : int          — tamaño de la grilla (N×N celdas)
    frac_vacia : float — fracción de celdas vacías (default 20%)
    seed : int       — semilla para reproducibilidad
    
    Retorna
    -------
    grilla : np.ndarray de shape (N, N) con valores 0, 1, 2
    """
    rng = np.random.default_rng(seed)
    n_celdas = N * N
    n_vacias  = int(n_celdas * frac_vacia)
    n_grupo_A = (n_celdas - n_vacias) // 2
    n_grupo_B =  n_celdas - n_vacias - n_grupo_A

    # Vector plano: 0=vacío, 1=A, 2=B
    flat = np.array([0] * n_vacias + [1] * n_grupo_A + [2] * n_grupo_B)
    rng.shuffle(flat)
    return flat.reshape(N, N)


def vecindad_moore(grilla: np.ndarray, fila: int, col: int):
    """
    Devuelve los valores de las 8 celdas vecinas (Moore) con bordes toroidales.
    El propio agente NO se incluye aquí — se suma aparte según la convención del TP.
    """
    N = grilla.shape[0]
    vecinos = []
    for df in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if df == 0 and dc == 0:
                continue
            vecinos.append(grilla[(fila + df) % N, (col + dc) % N])
    return vecinos  # lista de 8 valores


def celdas_vacias(grilla: np.ndarray):
    """Retorna lista de (fila, col) de celdas vacías."""
    filas, cols = np.where(grilla == 0)
    return list(zip(filas.tolist(), cols.tolist()))


def celdas_por_tipo(grilla: np.ndarray, tipo: int):
    """Retorna lista de (fila, col) de celdas con un tipo dado (1 o 2)."""
    filas, cols = np.where(grilla == tipo)
    return list(zip(filas.tolist(), cols.tolist()))
