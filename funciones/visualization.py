"""
visualization.py — Visualización para el modelo de Schelling.

- plot_snapshots : grilla 2×3 con los 6 estados de evolución
- plot_convergencia : curvas de descontentos e índice de segregación
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# Paleta de colores fiel al enunciado (naranja = A, verde = B, blanco = vacío)
CMAP = mcolors.ListedColormap(["#FFFFFF", "#E07B22", "#2E8B3A"])
NORM = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], CMAP.N)


def plot_snapshots(grillas: list, etiquetas: list, titulo: str = "",
                   figsize=(14, 9), guardar_como: str = None):
    """
    Genera un panel 2×3 con los 6 snapshots de evolución.
    
    Parámetros
    ----------
    grillas   : lista de arrays (mínimo 6)
    etiquetas : lista de strings con títulos
    titulo    : supertítulo de la figura
    guardar_como : path para guardar (None = solo mostrar)
    """
    n = min(len(grillas), 6)
    # Si hay más de 6, seleccionar 6 representativos
    if len(grillas) > 6:
        indices = np.linspace(0, len(grillas) - 1, 6, dtype=int)
        grillas  = [grillas[i]  for i in indices]
        etiquetas = [etiquetas[i] for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            g = grillas[idx]
            im = ax.imshow(g, cmap=CMAP, norm=NORM, interpolation="nearest",
                           origin="upper")
            ax.set_title(etiquetas[idx], color="white", fontsize=11, pad=6)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # Leyenda
    patch_A = mpatches.Patch(color="#E07B22", label="Grupo A")
    patch_B = mpatches.Patch(color="#2E8B3A", label="Grupo B")
    patch_V = mpatches.Patch(color="#FFFFFF", label="Vacío")
    fig.legend(handles=[patch_A, patch_B, patch_V],
               loc="lower center", ncol=3,
               fontsize=11, framealpha=0.3,
               labelcolor="white", facecolor="#2a2a4e",
               edgecolor="#555577")

    if titulo:
        fig.suptitle(titulo, color="white", fontsize=14, y=1.01, fontweight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if guardar_como:
        plt.savefig(guardar_como, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


def plot_convergencia(historial_desc: list, titulo: str = "",
                      guardar_como: str = None):
    """
    Curva de fracción de descontentos a lo largo de la simulación.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    x = np.arange(len(historial_desc))
    ax.plot(x, historial_desc, color="#E07B22", linewidth=2, label="Fracción descontentos")
    ax.fill_between(x, historial_desc, alpha=0.15, color="#E07B22")

    ax.set_xlabel("Tiempo (× 500 intentos)", color="white", fontsize=11)
    ax.set_ylabel("Fracción descontentos", color="white", fontsize=11)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color="#333355", linewidth=0.5)

    if titulo:
        ax.set_title(titulo, color="white", fontsize=13, pad=8)

    ax.legend(labelcolor="white", facecolor="#2a2a4e", edgecolor="#555577")
    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


def plot_comparacion_metricas(snaps_d1: list, labels_d1: list,
                               snaps_d2: list, labels_d2: list,
                               guardar_como: str = None):
    """
    Panel comparativo de índice de similitud entre Dinámica 1 y Dinámica 2.
    """
    from metrics import indice_similitud, fraccion_descontentos_d1, fraccion_descontentos_d2

    sim_d1 = [indice_similitud(g) for g in snaps_d1]
    sim_d2 = [indice_similitud(g) for g in snaps_d2]
    desc_d1 = [fraccion_descontentos_d1(g) for g in snaps_d1]
    desc_d2 = [fraccion_descontentos_d2(g) for g in snaps_d2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, vals_d1, vals_d2, ylabel in zip(
        [ax1, ax2],
        [sim_d1, desc_d1], [sim_d2, desc_d2],
        ["Índice de similitud", "Fracción de descontentos"]
    ):
        ax.set_facecolor("#1a1a2e")
        x = range(len(vals_d1))
        ax.plot(x, vals_d1, "o-", color="#E07B22", label="Dinámica 1 (minoría)")
        ax.plot(x, vals_d2, "s--", color="#2E8B3A", label="Dinámica 2 (mayoría)")
        ax.set_xticks(range(len(labels_d1)))
        ax.set_xticklabels(labels_d1, rotation=30, ha="right", fontsize=7, color="white")
        ax.set_ylabel(ylabel, color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.grid(True, color="#333355", linewidth=0.5)
        ax.legend(labelcolor="white", facecolor="#2a2a4e", edgecolor="#555577", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    fig.suptitle("Comparación de métricas: Dinámica 1 vs Dinámica 2",
                 color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
