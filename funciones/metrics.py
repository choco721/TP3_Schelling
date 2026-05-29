"""
metrics.py — Métricas para el modelo de Schelling.

- Índice de similitud local (fracción media de vecinos iguales)
- Fracción de agentes descontentos
- Detección de convergencia / frustración
"""

import numpy as np
from grid import vecindad_moore


def indice_similitud(grilla: np.ndarray) -> float:
    """
    Índice de segregación: fracción promedio de vecinos del mismo tipo,
    promediada sobre todos los agentes (celdas no vacías).
    
    Rango: 0 (completamente integrado) → 1 (completamente segregado).
    En una distribución aleatoria uniforme ≈ 0.5.
    """
    N = grilla.shape[0]
    fracciones = []
    for f in range(N):
        for c in range(N):
            tipo = grilla[f, c]
            if tipo == 0:
                continue
            vecinos = vecindad_moore(grilla, f, c)
            # Solo vecinos no vacíos
            vecinos_no_vacios = [v for v in vecinos if v != 0]
            if len(vecinos_no_vacios) == 0:
                continue
            frac = sum(1 for v in vecinos_no_vacios if v == tipo) / len(vecinos_no_vacios)
            fracciones.append(frac)
    return float(np.mean(fracciones)) if fracciones else 0.0


def fraccion_descontentos_d1(grilla: np.ndarray) -> float:
    """
    Fracción de agentes descontentos bajo Dinámica 1 (quieren ser minoría).
    """
    from dynamics import esta_contento_d1
    N = grilla.shape[0]
    desc = sum(1 for f in range(N) for c in range(N)
               if grilla[f, c] != 0 and not esta_contento_d1(grilla, f, c))
    total = np.sum(grilla > 0)
    return desc / total if total > 0 else 0.0


def fraccion_descontentos_d2(grilla: np.ndarray, umbral: float = 3/8) -> float:
    """
    Fracción de agentes descontentos bajo Dinámica 2 (quieren ser mayoría).
    """
    from dynamics import esta_contento_d2
    N = grilla.shape[0]
    desc = sum(1 for f in range(N) for c in range(N)
               if grilla[f, c] != 0 and not esta_contento_d2(grilla, f, c, umbral))
    total = np.sum(grilla > 0)
    return desc / total if total > 0 else 0.0


def calcular_historial_metricas(grillas: list, dinamica: int = 1,
                                 umbral_d2: float = 3/8) -> dict:
    """
    Dado un conjunto de snapshots, calcula métricas en cada uno.
    
    Retorna dict con listas:
      - 'similitud'    : índice de segregación por snapshot
      - 'descontentos' : fracción de descontentos por snapshot
    """
    sims = []
    descs = []
    for g in grillas:
        sims.append(indice_similitud(g))
        if dinamica == 1:
            descs.append(fraccion_descontentos_d1(g))
        else:
            descs.append(fraccion_descontentos_d2(g, umbral_d2))
    return {"similitud": sims, "descontentos": descs}


def es_sistema_frustrado(grilla_final: np.ndarray, dinamica: int = 1,
                          umbral_d2: float = 3/8,
                          umbral_frustracion: float = 0.05) -> bool:
    """
    Determina si el sistema terminó en estado frustrado.
    Un sistema está frustrado si la fracción de descontentos en el
    estado final supera el umbral de frustración.
    """
    if dinamica == 1:
        frac = fraccion_descontentos_d1(grilla_final)
    else:
        frac = fraccion_descontentos_d2(grilla_final, umbral_d2)
    return frac > umbral_frustracion
