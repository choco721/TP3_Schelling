"""
ising.py — Modelo de Ising 2D con algoritmo Metropolis-Hastings (MCMC).

El modelo de Ising es el equivalente físico del modelo de Schelling:
  - Espines ±1  ↔  Agentes tipo A/B
  - Temperatura T  ↔  Nivel de ruido / tolerancia
  - Interacción ferromagnética (J > 0)  ↔  Dinámica 2 (prefiero iguales)
  - Temperatura crítica Tc ≈ 2.269J/k_B  ↔  umbral de segregación

Hamiltoniano:  H = -J Σ_{<ij>} s_i · s_j
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ───────────────────────────────────────────────
# INICIALIZACIÓN
# ───────────────────────────────────────────────

def inicializar_ising(N: int, T: float, seed: int = 42) -> np.ndarray:
    """
    Inicializa una grilla N×N de espines ±1.
    Si T < 2.0 (régimen frío) → todos alineados (orden inicial).
    Si T >= 2.0               → distribución aleatoria.
    """
    rng = np.random.default_rng(seed)
    if T < 2.0:
        # Régimen ordenado: arrancar con todos los espines alineados
        return np.ones((N, N), dtype=int)
    else:
        # Régimen desordenado: arrancar aleatorio
        return rng.choice([-1, 1], size=(N, N)).astype(int)


# ───────────────────────────────────────────────
# ENERGÍA LOCAL
# ───────────────────────────────────────────────

def energia_local(grilla: np.ndarray, i: int, j: int, J: float = 1.0) -> float:
    """
    Energía de interacción del espín (i,j) con sus 4 vecinos (von Neumann).
    Bordes periódicos (toroidal).
    
    E_local = -J * s_ij * (s_arriba + s_abajo + s_izq + s_der)
    """
    N = grilla.shape[0]
    s = grilla[i, j]
    vecinos_suma = (grilla[(i-1) % N, j] +
                    grilla[(i+1) % N, j] +
                    grilla[i, (j-1) % N] +
                    grilla[i, (j+1) % N])
    return -J * s * vecinos_suma


def energia_total(grilla: np.ndarray, J: float = 1.0) -> float:
    """Energía total del sistema."""
    N = grilla.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(N):
            # Dividimos por 2 para no contar cada par dos veces
            E += -J * grilla[i, j] * (grilla[(i+1) % N, j] + grilla[i, (j+1) % N])
    return E


# ───────────────────────────────────────────────
# PASO METROPOLIS
# ───────────────────────────────────────────────

def paso_metropolis(grilla: np.ndarray, T: float, J: float = 1.0,
                    rng=None) -> int:
    """
    Realiza N² intentos de flip (un barrido completo de la grilla).
    
    Criterio Metropolis:
      ΔE = 2 * E_local  (energía si flippeamos el espín)
      Si ΔE ≤ 0 → flip siempre (baja la energía)
      Si ΔE > 0 → flip con probabilidad exp(-ΔE / T)
    
    Retorna número de flips aceptados.
    """
    if rng is None:
        rng = np.random.default_rng()
    N = grilla.shape[0]
    flips = 0
    k_B = 1.0  # Unidades de k_B = 1

    for _ in range(N * N):
        i = rng.integers(N)
        j = rng.integers(N)
        
        # ΔE al flipear el espín (i,j)
        s = grilla[i, j]
        vecinos = (grilla[(i-1) % N, j] + grilla[(i+1) % N, j] +
                   grilla[i, (j-1) % N] + grilla[i, (j+1) % N])
        dE = 2 * J * s * vecinos

        # Criterio de aceptación Metropolis
        if dE <= 0 or rng.random() < np.exp(-dE / (k_B * T)):
            grilla[i, j] = -s
            flips += 1

    return flips


# ───────────────────────────────────────────────
# SIMULACIÓN COMPLETA
# ───────────────────────────────────────────────

def correr_ising(N: int = 50, T: float = 2.0, n_pasos: int = 1000,
                 n_equilibrio: int = 500, J: float = 1.0,
                 snapshots_en: list = None, seed: int = 42):
    """
    Corre la simulación de Ising y retorna snapshots y observables.
    
    Parámetros
    ----------
    N          : tamaño de grilla
    T          : temperatura (en unidades de J/k_B)
    n_pasos    : número de pasos Metropolis totales
    n_equilibrio: pasos de termalización (antes de medir)
    snapshots_en : pasos en que guardar snapshot
    
    Retorna
    -------
    grillas_snap : lista de arrays
    etiquetas    : lista de strings
    magnetizacion: lista de |M| por paso (post-equilibrio)
    energias     : lista de E/N² por paso (post-equilibrio)
    """
    rng = np.random.default_rng(seed)
    grilla = inicializar_ising(N, T, seed)

    if snapshots_en is None:
        pasos_post = n_pasos - n_equilibrio
        snapshots_en = [0] + list(np.linspace(n_equilibrio,
                                               n_pasos - 1,
                                               5, dtype=int))

    grillas_snap = []
    etiquetas = []
    magnetizacion = []
    energias = []

    for paso in range(n_pasos):
        paso_metropolis(grilla, T, J, rng)

        if paso in snapshots_en:
            grillas_snap.append(grilla.copy())
            etiquetas.append(f"Paso {paso}  (T={T:.2f})")

        if paso >= n_equilibrio:
            M = np.abs(np.mean(grilla))
            E = energia_total(grilla, J) / (N * N)
            magnetizacion.append(M)
            energias.append(E)

    return grillas_snap, etiquetas, magnetizacion, energias


# ───────────────────────────────────────────────
# TRANSICIÓN DE FASE: M vs T
# ───────────────────────────────────────────────

def curva_magnetizacion_vs_temperatura(N: int = 30, n_pasos: int = 600,
                                        n_equilibrio: int = 400,
                                        T_min: float = 0.5, T_max: float = 4.5,
                                        n_T: int = 20, seed: int = 42):
    """
    Calcula la magnetización media |<M>| en función de la temperatura.
    Permite visualizar la transición de fase en T_c ≈ 2.269.
    """
    Ts = np.linspace(T_min, T_max, n_T)
    Ms = []
    Es = []

    for T in Ts:
        _, _, mag, en = correr_ising(N=N, T=T, n_pasos=n_pasos,
                                      n_equilibrio=n_equilibrio, seed=seed)
        Ms.append(np.mean(mag))
        Es.append(np.mean(en))

    return Ts, Ms, Es


# ───────────────────────────────────────────────
# VISUALIZACIÓN
# ───────────────────────────────────────────────

CMAP_ISING = mcolors.ListedColormap(["#E07B22", "#2E8B3A"])  # -1=naranja, +1=verde
NORM_ISING = mcolors.BoundaryNorm([-1.5, 0, 1.5], CMAP_ISING.N)


def plot_ising_snapshots(grillas: list, etiquetas: list,
                          titulo: str = "", guardar_como: str = None):
    """Panel 2×3 de snapshots del modelo de Ising."""
    n = min(len(grillas), 6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            ax.imshow(grillas[idx], cmap=CMAP_ISING, norm=NORM_ISING,
                      interpolation="nearest")
            ax.set_title(etiquetas[idx], color="white", fontsize=10, pad=5)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    import matplotlib.patches as mpatches
    pA = mpatches.Patch(color="#E07B22", label="Espín −1")
    pB = mpatches.Patch(color="#2E8B3A", label="Espín +1")
    fig.legend(handles=[pA, pB], loc="lower center", ncol=2,
               fontsize=11, labelcolor="white",
               facecolor="#2a2a4e", edgecolor="#555577")

    if titulo:
        fig.suptitle(titulo, color="white", fontsize=14, y=1.01, fontweight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if guardar_como:
        plt.savefig(guardar_como, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


def plot_magnetizacion_vs_T(Ts, Ms, guardar_como: str = None):
    """Curva de magnetización vs temperatura con línea Tc."""
    Tc = 2.269
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.plot(Ts, Ms, "o-", color="#E07B22", linewidth=2, label="|⟨M⟩|")
    ax.axvline(Tc, color="#9999ff", linestyle="--", linewidth=1.5,
               label=f"$T_c \\approx {Tc}$")
    ax.fill_betweenx([0, 1], 0, Tc, alpha=0.05, color="#2E8B3A",
                     label="Fase ordenada")
    ax.fill_betweenx([0, 1], Tc, max(Ts), alpha=0.05, color="#E07B22",
                     label="Fase desordenada")

    ax.set_xlabel("Temperatura T  (en unidades J/k_B)", color="white", fontsize=11)
    ax.set_ylabel("|⟨M⟩| — Magnetización", color="white", fontsize=11)
    ax.set_xlim(min(Ts), max(Ts))
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="white")
    ax.grid(True, color="#333355", linewidth=0.5)
    ax.legend(labelcolor="white", facecolor="#2a2a4e", edgecolor="#555577", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.set_title("Transición de fase — Modelo de Ising 2D",
                 color="white", fontsize=13, pad=8)
    plt.tight_layout()
    if guardar_como:
        plt.savefig(guardar_como, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
