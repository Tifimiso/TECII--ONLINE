"""
hit_distribution.py
-------------------
Análise da distribuição de hits nos detetores.

Usa a TTree 'Hits' (uma entrada por hit) com as branches:
    detectorID, hitPosX_cm, hitPosY_cm, particleCharge, particlePDG, Edep_keV

Análises incluídas:
    - Distribuição de hits em X e Y para cada detetor.
    - Distribuição de hits em X e Y por tipo de partícula
      (carregada ou neutra) para cada detetor.
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot


OUTPUT_DIR = "plots/hits"

_SEARCH_DIRS = ["data", "../RootFiles", "ROOTFILES", "."]
_FILENAME_PATTERN = "AmberTarget_Run_*.root"


def _encontrar_ficheiros():
    """Procura os ficheiros ROOT nos diretórios habituais."""
    for d in _SEARCH_DIRS:
        matches = sorted(glob.glob(os.path.join(d, _FILENAME_PATTERN)))
        if matches:
            print(f"[OK] {len(matches)} ficheiro(s) ROOT encontrado(s) em '{d}/'")
            return matches
    print("[ERRO] Nenhum ficheiro AmberTarget_Run_*.root encontrado.")
    print(f"  Diretórios pesquisados: {_SEARCH_DIRS}")
    print("  Ver data/README.md para instruções.")
    sys.exit(1)


def _carregar_dados(ficheiros):
    """Carrega a TTree 'Hits' de todos os ficheiros ROOT."""
    branches = [
        "detectorID", "hitPosX_cm", "hitPosY_cm",
        "particleCharge", "particlePDG", "Edep_keV",
    ]
    arrays = []
    for fname in ficheiros:
        with uproot.open(fname) as f:
            if "Hits" not in [k.split(";")[0] for k in f.keys()]:
                print(f"  [AVISO] TTree 'Hits' não encontrada em {fname} — ignorado.")
                continue
            arr = f["Hits"].arrays(branches, library="np")
            n = len(arr["detectorID"])
            print(f"  [OK] {fname}: {n:,} hits.")
            arrays.append(arr)

    if not arrays:
        print("[ERRO] Nenhum ficheiro continha dados válidos em 'Hits'.")
        sys.exit(1)

    return {k: np.concatenate([a[k] for a in arrays]) for k in branches}


def plot_hit_distribution_xy(data):
    """Distribuição de hits em X e Y para cada detetor."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    det_id = data["detectorID"]
    x = data["hitPosX_cm"]
    y = data["hitPosY_cm"]

    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # --- Gráfico 1: projeções X e Y (grelha 2×4) ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=150)
    fig.suptitle("Distribuição espacial dos hits por detetor", fontsize=13, fontweight="bold")

    for i in range(4):
        mask = det_id == i
        xi, yi = x[mask], y[mask]

        ax_x = axes[0, i]
        ax_x.hist(xi, bins=120, color=cores[i], alpha=0.85,
                  histtype="stepfilled", edgecolor=cores[i], linewidth=0.6)
        ax_x.set_title(f"Detetor {i}  ({mask.sum():,} hits)", fontsize=10, fontweight="bold")
        ax_x.set_xlabel("Posição X (cm)", fontsize=9)
        ax_x.set_ylabel("Contagens", fontsize=9)
        ax_x.grid(True, linestyle="--", alpha=0.4)

        ax_y = axes[1, i]
        ax_y.hist(yi, bins=120, color=cores[i], alpha=0.85,
                  histtype="stepfilled", edgecolor=cores[i], linewidth=0.6)
        ax_y.set_xlabel("Posição Y (cm)", fontsize=9)
        ax_y.set_ylabel("Contagens", fontsize=9)
        ax_y.grid(True, linestyle="--", alpha=0.4)

    axes[0, 0].annotate("Proj. X", xy=(-0.38, 0.5), xycoords="axes fraction",
                        va="center", rotation=90, fontsize=10, fontweight="bold")
    axes[1, 0].annotate("Proj. Y", xy=(-0.38, 0.5), xycoords="axes fraction",
                        va="center", rotation=90, fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hits_xy_por_detetor.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] plot_hit_distribution_xy → {out}")

    # --- Gráfico 2: mapas 2D X vs Y ---
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
    fig2.suptitle("Mapa 2D de hits (X vs Y) por detetor", fontsize=13, fontweight="bold")

    for i in range(4):
        mask = det_id == i
        xi, yi = x[mask], y[mask]
        ax = axes2[i]
        h, xedges, yedges = np.histogram2d(xi, yi, bins=100)
        im = ax.imshow(
            h.T, origin="lower", aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="hot_r", interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Detetor {i}", fontsize=11, fontweight="bold")
        ax.set_xlabel("X (cm)", fontsize=9)
        ax.set_ylabel("Y (cm)", fontsize=9)

    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "hits_mapa2d_por_detetor.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] plot_hit_distribution_xy (2D) → {out2}")


def plot_hit_distribution_by_charge(data):
    """Distribuição de hits em X e Y por tipo de partícula (carregada ou neutra)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    det_id = data["detectorID"]
    x = data["hitPosX_cm"]
    y = data["hitPosY_cm"]
    charge = data["particleCharge"]

    carregado = charge != 0
    neutro = charge == 0

    n_carregado = carregado.sum()
    n_neutro = neutro.sum()
    print(f"  Hits carregados: {n_carregado:,} ({100*n_carregado/(n_carregado+n_neutro):.1f}%)")
    print(f"  Hits neutros:    {n_neutro:,} ({100*n_neutro/(n_carregado+n_neutro):.1f}%)")

    # --- Painéis por detetor ---
    for i in range(4):
        det_mask = det_id == i

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        fig.suptitle(f"Distribuição de hits por carga — Detetor {i}", fontsize=12, fontweight="bold")

        for ax, coord, label_eixo in zip(axes, [x, y], ["X (cm)", "Y (cm)"]):
            for nome, mask, cor in [
                ("Carregadas", det_mask & carregado, "#1f77b4"),
                ("Neutras",    det_mask & neutro,    "#ff7f0e"),
            ]:
                vals = coord[mask]
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=120, histtype="stepfilled", alpha=0.65,
                        label=f"{nome} ({len(vals):,})",
                        color=cor, edgecolor=cor, linewidth=0.6)
            ax.set_xlabel(label_eixo, fontsize=10)
            ax.set_ylabel("Contagens", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"hits_por_carga_detector{i}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] plot_hit_distribution_by_charge — detetor {i} → {out}")

    # --- Resumo: fracção carregadas vs neutras por detetor ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fracs_c, fracs_n, totais = [], [], []
    for i in range(4):
        m = det_id == i
        tot = m.sum()
        totais.append(tot)
        fracs_c.append((m & carregado).sum() / tot * 100 if tot > 0 else 0)
        fracs_n.append((m & neutro).sum() / tot * 100 if tot > 0 else 0)

    x_pos = np.arange(4)
    w = 0.35
    ax.bar(x_pos - w / 2, fracs_c, w, label="Carregadas", color="#1f77b4", alpha=0.85)
    ax.bar(x_pos + w / 2, fracs_n, w, label="Neutras", color="#ff7f0e", alpha=0.85)
    for i, (fc, fn, tot) in enumerate(zip(fracs_c, fracs_n, totais)):
        ax.text(i - w / 2, fc + 0.5, f"{fc:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, fn + 0.5, f"{fn:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Número do detetor", fontsize=11)
    ax.set_ylabel("Fracção de hits (%)", fontsize=11)
    ax.set_title("Fracção de hits carregados vs. neutros por detetor", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Det. {i}\n({totais[i]:,})" for i in range(4)], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hits_fracao_carga_por_detetor.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] plot_hit_distribution_by_charge (resumo) → {out}")


def main():
    ficheiros = _encontrar_ficheiros()
    data = _carregar_dados(ficheiros)
    n_total = len(data["detectorID"])
    print(f"[OK] {n_total:,} hits no total (4 runs combinados).")

    plot_hit_distribution_xy(data)
    plot_hit_distribution_by_charge(data)

    print(f"\n[OK] Gráficos guardados em {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
