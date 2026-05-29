"""
dynamics.py — Reglas de satisfacción y pasos de simulación.

Dinámica 1: agente CONTENTO si es MINORÍA en su vecindad (quiere ser diferente)
Dinámica 2: agente CONTENTO si es MAYORÍA en su vecindad (Schelling original)

Convención del TP: el recuento de vecindad INCLUYE al propio agente.
Mecánica: se selecciona un par de agentes de DISTINTO color al azar.
          Si AMBOS están descontentos → intercambian posiciones.
          Las actualizaciones son asincrónicas.
"""

import numpy as np
from grid import vecindad_moore


# ───────────────────────────────────────────────
# EVALUACIÓN DE SATISFACCIÓN
# ───────────────────────────────────────────────

def fraccion_similares(grilla: np.ndarray, fila: int, col: int) -> float:
    """
    Fracción de vecinos (incluyendo al agente mismo) que son del mismo tipo.
    Solo considera las 8 celdas + el propio agente (ignora celdas vacías vecinas
    al calcular la fracción, pero las incluye en el conteo total = 9).
    
    Convención del TP: el agente se incluye en el recuento.
    """
    tipo = grilla[fila, col]
    vecinos = vecindad_moore(grilla, fila, col)
    # Incluir al propio agente en el recuento (convención del TP)
    todos = vecinos + [tipo]
    n_iguales = sum(1 for v in todos if v == tipo)
    n_total = len(todos)  # siempre 9
    return n_iguales / n_total


def esta_contento_d1(grilla: np.ndarray, fila: int, col: int) -> bool:
    """
    Dinámica 1 — contento = ser MINORÍA (fracción similares < 0.5).
    Un agente es minoría cuando menos de la mitad de su vecindad (incluyéndose)
    comparte su color.
    """
    return fraccion_similares(grilla, fila, col) < 0.5


def esta_contento_d2(grilla: np.ndarray, fila: int, col: int,
                     umbral: float = 3/8) -> bool:
    """
    Dinámica 2 — contento = ser MAYORÍA (fracción similares >= umbral).
    Umbral clásico de Schelling: 3/8 ≈ 0.375  (quiero que al menos 3 de 8 sean iguales).
    El TP usa 0.5 por defecto (mayoría estricta).
    """
    return fraccion_similares(grilla, fila, col) >= umbral


# ───────────────────────────────────────────────
# PASO DE SIMULACIÓN
# ───────────────────────────────────────────────

def paso_simulacion(grilla: np.ndarray, dinamica: int = 1,
                    umbral_d2: float = 3/8, rng=None) -> tuple[int, int]:
    """
    Ejecuta UN intento de intercambio:
    1. Selecciona al azar un par de agentes de distinto color.
    2. Evalúa satisfacción de ambos según la dinámica elegida.
    3. Si AMBOS están descontentos → intercambian posiciones.
    
    Retorna: (n_intercambios_realizados, tipo_de_par_seleccionado)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = grilla.shape[0]
    
    # Obtener posiciones de tipo 1 y tipo 2
    pos1 = list(zip(*np.where(grilla == 1)))
    pos2 = list(zip(*np.where(grilla == 2)))
    
    if len(pos1) == 0 or len(pos2) == 0:
        return 0, 0

    # Seleccionar un agente de cada grupo al azar
    idx1 = rng.integers(len(pos1))
    idx2 = rng.integers(len(pos2))
    f1, c1 = pos1[idx1]
    f2, c2 = pos2[idx2]

    # Evaluar satisfacción según dinámica
    if dinamica == 1:
        desc1 = not esta_contento_d1(grilla, f1, c1)
        desc2 = not esta_contento_d1(grilla, f2, c2)
    else:  # dinamica == 2
        desc1 = not esta_contento_d2(grilla, f1, c1, umbral_d2)
        desc2 = not esta_contento_d2(grilla, f2, c2, umbral_d2)

    # Solo intercambian si AMBOS están descontentos
    if desc1 and desc2:
        grilla[f1, c1], grilla[f2, c2] = grilla[f2, c2], grilla[f1, c1]
        return 1, 1

    return 0, 1


def correr_simulacion(grilla_inicial: np.ndarray, dinamica: int = 1,
                      n_intentos_max: int = 500_000,
                      intentos_sin_swap_para_detener: int = 10_000,
                      snapshots_en: list = None,
                      umbral_d2: float = 3/8,
                      seed: int = 0) -> tuple[list, list, list]:
    """
    Corre la simulación completa y guarda snapshots en los pasos indicados.
    
    Parámetros
    ----------
    grilla_inicial : ndarray       — estado inicial
    dinamica : 1 o 2               — dinámica a usar
    n_intentos_max : int           — máximo de intentos totales
    intentos_sin_swap_para_detener — se detiene si no hay swap en N intentos consecutivos
    snapshots_en : list[int]       — lista de número de SWAPS en que guardar snapshot
                                     Si None, se guarda cada ~n_intentos_max/5 swaps
    umbral_d2 : float              — umbral para dinámica 2
    seed : int                     — semilla RNG
    
    Retorna
    -------
    grillas_snapshot : list de ndarray (copias de la grilla en cada snapshot)
    etiquetas        : list de str (etiquetas para cada snapshot)
    historial_desc   : list de float (fracción de descontentos por swap)
    """
    rng = np.random.default_rng(seed)
    grilla = grilla_inicial.copy()
    N = grilla.shape[0]
    n_agentes = np.sum(grilla > 0)

    # Snapshots automáticos si no se especifican
    if snapshots_en is None:
        snapshots_en = [0, 50, 200, 500, 1500, -1]  # -1 = estado final

    grillas_snapshot = []
    etiquetas = []
    historial_desc = []

    # Snapshot inicial (swap 0)
    grillas_snapshot.append(grilla.copy())
    etiquetas.append("Estado inicial")

    swaps_totales = 0
    intentos_sin_swap = 0
    swaps_siguientes_snapshot = [s for s in snapshots_en if s > 0]
    idx_snap = 0

    for intento in range(n_intentos_max):
        n_swap, _ = paso_simulacion(grilla, dinamica, umbral_d2, rng)
        
        if n_swap > 0:
            swaps_totales += n_swap
            intentos_sin_swap = 0
        else:
            intentos_sin_swap += 1

        # Guardar snapshot en los swaps indicados
        if idx_snap < len(swaps_siguientes_snapshot):
            if swaps_totales >= swaps_siguientes_snapshot[idx_snap]:
                grillas_snapshot.append(grilla.copy())
                etiquetas.append(f"{swaps_totales:,} mudanzas")
                idx_snap += 1

        # Registrar fracción de descontentos periódicamente (cada 500 intentos)
        if intento % 500 == 0:
            frac = calcular_fraccion_descontentos(grilla, dinamica, umbral_d2)
            historial_desc.append(frac)

        # Criterio de parada
        if intentos_sin_swap >= intentos_sin_swap_para_detener:
            break

    # Snapshot final (estado estacionario o cuasi-estacionario)
    if len(grillas_snapshot) < 6:
        grillas_snapshot.append(grilla.copy())
        etiquetas.append(f"Final ({swaps_totales:,} mudanzas)")
    elif etiquetas[-1] != f"Final ({swaps_totales:,} mudanzas)":
        # Reemplazar el último snapshot por el estado real final
        grillas_snapshot[-1] = grilla.copy()
        etiquetas[-1] = f"Final ({swaps_totales:,} mudanzas)"

    return grillas_snapshot, etiquetas, historial_desc


def calcular_fraccion_descontentos(grilla: np.ndarray, dinamica: int = 1,
                                    umbral_d2: float = 3/8) -> float:
    """
    Calcula la fracción de agentes (no vacíos) que están descontentos.
    """
    N = grilla.shape[0]
    n_descontentos = 0
    n_agentes = 0
    for f in range(N):
        for c in range(N):
            tipo = grilla[f, c]
            if tipo == 0:
                continue
            n_agentes += 1
            if dinamica == 1:
                if not esta_contento_d1(grilla, f, c):
                    n_descontentos += 1
            else:
                if not esta_contento_d2(grilla, f, c, umbral_d2):
                    n_descontentos += 1
    return n_descontentos / n_agentes if n_agentes > 0 else 0.0
