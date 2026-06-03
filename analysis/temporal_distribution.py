"""
temporal_distribution.py
------------------------
Análise da distribuição temporal dos hits nos detetores.

Usa a TTree 'Hits' (uma entrada por hit) com as branches:
    detectorID, particleHitTime_ns, particleCharge, particlePDG

Análises incluídas:
    - Histograma da distribuição temporal dos hits por detetor.
    - Sobreposição dos 4 detetores (estrutura temporal do feixe).
    - Comparação entre espécies de partículas (detetor 0).
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
    branches = ["detectorID", "particleHitTime_ns", "particleCharge", "particlePDG"]
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


def plot_temporal_distribution(data):
    """Histograma da distribuição temporal dos hits por detetor."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    det_id = data["detectorID"]
    t = data["particleHitTime_ns"]
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Limites globais (excluir zeros/negativos — partículas sem hit)
    t_valid = t[t > 0]
    if len(t_valid) == 0:
        print("[AVISO] Nenhum tempo > 0 encontrado.")
        return
    t_min, t_max = t_valid.min(), np.percentile(t_valid, 99.5)
    bins = np.linspace(t_min, t_max, 200)

    # --- Gráfico 1: sobreposição dos 4 detetores ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for i, cor in zip(range(4), cores):
        mask = (det_id == i) & (t > 0)
        ti = t[mask]
        if len(ti) == 0:
            continue
        ax.hist(ti, bins=bins, histtype="step", linewidth=1.8,
                color=cor, label=f"Detetor {i}  ({len(ti):,} hits)")

    ax.set_xlabel("Tempo de chegada (ns)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Contagens", fontsize=12, fontweight="bold")
    ax.set_title("Distribuição temporal dos hits por detetor", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "temporal_por_detetor.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] plot_temporal_distribution (overlay) → {out}")

    # --- Gráfico 2: painéis individuais com escala logarítmica ---
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
    fig2.suptitle("Distribuição temporal dos hits — detetores individuais (escala log)",
                  fontsize=12, fontweight="bold")

    for i, (ax2, cor) in enumerate(zip(axes2, cores)):
        mask = (det_id == i) & (t > 0)
        ti = t[mask]
        if len(ti) == 0:
            ax2.text(0.5, 0.5, "Sem dados", transform=ax2.transAxes, ha="center")
            continue
        # bins adaptados ao intervalo do detetor
        ti_bins = np.linspace(ti.min(), np.percentile(ti, 99.5), 120)
        ax2.hist(ti, bins=ti_bins, color=cor, alpha=0.85,
                 histtype="stepfilled", edgecolor=cor, linewidth=0.6)
        ax2.set_yscale("log")
        ax2.set_title(f"Detetor {i}  ({len(ti):,})", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Tempo (ns)", fontsize=9)
        ax2.set_ylabel("Contagens", fontsize=9)
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "temporal_individual_log.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] plot_temporal_distribution (painéis log) → {out2}")

    # --- Gráfico 3: carregadas vs neutras (detetor 0) ---
    charge = data["particleCharge"]
    det0_mask = (det_id == 0) & (t > 0)
    t0 = t[det0_mask]
    c0 = charge[det0_mask]

    if len(t0) > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 5), dpi=150)
        bins3 = np.linspace(t0.min(), np.percentile(t0, 99.5), 150)
        for nome, mask_c, cor in [
            ("Carregadas", c0 != 0, "#1f77b4"),
            ("Neutras",    c0 == 0, "#ff7f0e"),
        ]:
            vals = t0[mask_c]
            if len(vals) == 0:
                continue
            ax3.hist(vals, bins=bins3, histtype="stepfilled", alpha=0.65,
                     label=f"{nome} ({len(vals):,})", color=cor,
                     edgecolor=cor, linewidth=0.6)
        ax3.set_xlabel("Tempo de chegada (ns)", fontsize=11)
        ax3.set_ylabel("Contagens", fontsize=11)
        ax3.set_title("Distribuição temporal: carregadas vs. neutras — Detetor 0",
                      fontsize=11, fontweight="bold")
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        out3 = os.path.join(OUTPUT_DIR, "temporal_carga_detetor0.png")
        plt.savefig(out3, dpi=150)
        plt.close()
        print(f"[OK] plot_temporal_distribution (carga detetor 0) → {out3}")


def plot_temporal_by_species(data):
    """Distribuição temporal por espécie de partícula — detetor 0."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    det_id = data["detectorID"]
    t = data["particleHitTime_ns"]
    pdg = data["particlePDG"]

    det0_mask = (det_id == 0) & (t > 0)
    t0 = t[det0_mask]
    pdg0 = pdg[det0_mask]

    if len(t0) == 0:
        print("[AVISO] Nenhum dado de tempo > 0 no detetor 0.")
        return

    especies = [
        (r"$e^{\pm}$",   [11, -11],   "#ff7f0e"),
        (r"$\gamma$",    [22],         "#2ca02c"),
        (r"$\pi^{\pm}$", [211, -211],  "#d62728"),
        (r"$p$",         [2212],        "#9467bd"),
        (r"$K^{\pm}$",   [321, -321],  "#8c564b"),
        (r"$\mu^{\pm}$", [13, -13],    "#1f77b4"),
    ]

    bins = np.linspace(t0.min(), np.percentile(t0, 99.5), 150)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for nome, pdg_list, cor in especies:
        mask = np.isin(pdg0, pdg_list)
        vals = t0[mask]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.6,
                color=cor, label=f"{nome}  ({len(vals):,})")

    ax.set_xlabel("Tempo de chegada (ns)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Contagens", fontsize=12, fontweight="bold")
    ax.set_title("Distribuição temporal por espécie — Detetor 0", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "temporal_por_especie_detetor0.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] plot_temporal_by_species → {out}")


def main():
    ficheiros = _encontrar_ficheiros()
    data = _carregar_dados(ficheiros)
    n_total = len(data["detectorID"])
    print(f"[OK] {n_total:,} hits no total (4 runs combinados).")

    plot_temporal_distribution(data)
    plot_temporal_by_species(data)

    print(f"\n[OK] Gráficos guardados em {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
