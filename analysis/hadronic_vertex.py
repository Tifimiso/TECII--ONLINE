"""
hadronic_vertex.py
------------------
Análise dos vértices hadrónicos primários e secundários.

Lê a TTree 'hadronicVertex' dos ficheiros AmberTarget_Run_*.root (pasta data/)
com as branches vertexPosZ_cm e IsPrimary, e produz três figuras em
plots/vertex/:
    1. vertex_z_overlay.png        — posição Z de todos os vértices hadrónicos,
                                      primários (IsPrimary==1) e secundários
                                      (IsPrimary==0) em sobreposição.
    2. vertex_z_individual.png     — painéis individuais para primários e
                                      secundários, em escala logarítmica.
    3. vertex_contagem_por_run.png — gráfico de barras com o número de vértices
                                      primários vs secundários por run.
"""

import os
import re
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot


TREE_NAME = "hadronicVertex"
BRANCHES = ["vertexPosZ_cm", "IsPrimary"]
OUTPUT_DIR = os.path.join("plots", "vertex")
SEARCH_DIRS = ["data", "../RootFiles", "."]

# Cores consistentes com os restantes scripts do projecto
COLOR_ALL = "#1f77b4"        # Azul Cobalto
COLOR_PRIMARY = "#2ca02c"    # Verde Esmeralda
COLOR_SECONDARY = "#d62728"  # Vermelho Carmim

# Limites em Z (cm) — a simulação cobre aproximadamente -500 a +500 cm
Z_MIN, Z_MAX = -500.0, 500.0
N_BINS = 100


def _num_run(path):
    """Extrai o número da run do nome do ficheiro AmberTarget_Run_<N>.root."""
    m = re.search(r"AmberTarget_Run_(\d+)\.root$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _encontrar_ficheiros():
    """
    Procura os ficheiros AmberTarget_Run_*.root nas pastas habituais
    (por ordem: data/, ../RootFiles/, .) e devolve-os ordenados por número de run.
    """
    for s_dir in SEARCH_DIRS:
        matches = glob.glob(os.path.join(s_dir, "AmberTarget_Run_*.root"))
        if matches:
            return sorted(matches, key=_num_run)

    raise FileNotFoundError(
        "Nenhum ficheiro ROOT de simulação (AmberTarget_Run_*.root) foi "
        "encontrado nas pastas: {}".format(SEARCH_DIRS)
    )


def _carregar_dados(data=None):
    """
    Carrega as branches vertexPosZ_cm e IsPrimary da TTree 'hadronicVertex'.

    Aceita um dicionário já carregado (devolvido tal e qual), um caminho para
    ficheiro ou pasta, uma lista de ficheiros, ou None (procura automática nas
    pastas habituais).

    Devolve um dicionário com as arrays concatenadas de todas as runs, mais
    uma array 'run' que identifica a run de origem de cada vértice (necessária
    para a contagem por run).
    """
    if isinstance(data, dict):
        return data

    if isinstance(data, str):
        if os.path.isdir(data):
            ficheiros = sorted(
                glob.glob(os.path.join(data, "AmberTarget_Run_*.root")),
                key=_num_run,
            )
        elif os.path.isfile(data):
            ficheiros = [data]
        else:
            ficheiros = _encontrar_ficheiros()
    elif isinstance(data, (list, tuple)):
        ficheiros = list(data)
    else:
        ficheiros = _encontrar_ficheiros()

    if not ficheiros:
        raise FileNotFoundError(
            "Nenhum ficheiro ROOT de simulação (AmberTarget_Run_*.root) foi "
            "encontrado nas pastas: {}".format(SEARCH_DIRS)
        )

    print(f"A carregar dados de: {ficheiros}")

    z_parts, p_parts, run_parts = [], [], []
    for path in ficheiros:
        run_id = _num_run(path)
        arrs = uproot.open(f"{path}:{TREE_NAME}").arrays(BRANCHES, library="np")
        z = arrs["vertexPosZ_cm"]
        p = arrs["IsPrimary"]
        z_parts.append(z)
        p_parts.append(p)
        run_parts.append(np.full(z.shape, run_id, dtype=int))

    return {
        "vertexPosZ_cm": np.concatenate(z_parts),
        "IsPrimary": np.concatenate(p_parts),
        "run": np.concatenate(run_parts),
    }


def plot_vertex_z_overlay(data=None):
    """
    Histograma da posição Z de todos os vértices hadrónicos, com primários
    (IsPrimary==1) e secundários (IsPrimary==0) sobrepostos.
    Guarda o gráfico em plots/vertex/vertex_z_overlay.png.
    """
    dados = _carregar_dados(data)
    z = dados["vertexPosZ_cm"]
    is_primary = dados["IsPrimary"]

    z_primary = z[is_primary == 1]
    z_secondary = z[is_primary == 0]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    bins = np.linspace(Z_MIN, Z_MAX, N_BINS)

    # contorno para o total, preenchimento translúcido para cada categoria
    plt.hist(
        z,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        color=COLOR_ALL,
        label=f"Todos os vértices (N = {z.size:,})",
    )
    plt.hist(
        z_primary,
        bins=bins,
        histtype="stepfilled",
        alpha=0.6,
        color=COLOR_PRIMARY,
        edgecolor=COLOR_PRIMARY,
        linewidth=1.5,
        label=f"Primários — IsPrimary = 1 (N = {z_primary.size:,})",
    )
    plt.hist(
        z_secondary,
        bins=bins,
        histtype="stepfilled",
        alpha=0.45,
        color=COLOR_SECONDARY,
        edgecolor=COLOR_SECONDARY,
        linewidth=1.5,
        label=f"Secundários — IsPrimary = 0 (N = {z_secondary.size:,})",
    )

    plt.xlabel("Posição Z do vértice [cm]", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Número de vértices hadrónicos", fontsize=12, fontweight="bold", labelpad=10)
    plt.title("Distribuição em Z dos vértices hadrónicos", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=10, loc="upper left")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "vertex_z_overlay.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {out_path}")


def plot_vertex_z_individual(data=None):
    """
    Painéis individuais da posição Z dos vértices primários e secundários,
    ambos em escala logarítmica no eixo Y (a cauda da distribuição abrange
    várias ordens de grandeza).
    Guarda o gráfico em plots/vertex/vertex_z_individual.png.
    """
    dados = _carregar_dados(data)
    z = dados["vertexPosZ_cm"]
    is_primary = dados["IsPrimary"]

    z_primary = z[is_primary == 1]
    z_secondary = z[is_primary == 0]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    bins = np.linspace(Z_MIN, Z_MAX, N_BINS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharex=True)

    ax1.hist(
        z_primary,
        bins=bins,
        histtype="stepfilled",
        alpha=0.75,
        color=COLOR_PRIMARY,
        edgecolor=COLOR_PRIMARY,
        linewidth=1.5,
    )
    ax1.set_yscale("log")
    ax1.set_title(f"Vértices primários (IsPrimary = 1) — N = {z_primary.size:,}",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Posição Z do vértice [cm]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Número de vértices (escala log)", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2.hist(
        z_secondary,
        bins=bins,
        histtype="stepfilled",
        alpha=0.6,
        color=COLOR_SECONDARY,
        edgecolor=COLOR_SECONDARY,
        linewidth=1.5,
    )
    ax2.set_yscale("log")
    ax2.set_title(f"Vértices secundários (IsPrimary = 0) — N = {z_secondary.size:,}",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Posição Z do vértice [cm]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Número de vértices (escala log)", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.suptitle("Posição Z dos vértices hadrónicos — primários vs secundários",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "vertex_z_individual.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {out_path}")


def plot_contagem_por_run(data=None):
    """
    Gráfico de barras com o número de vértices hadrónicos primários vs
    secundários em cada run. O eixo Y está em escala logarítmica, pois os
    vértices secundários são uma ordem de grandeza mais abundantes que os
    primários. As contagens exactas são anotadas no topo de cada barra.
    Guarda o gráfico em plots/vertex/vertex_contagem_por_run.png.
    """
    dados = _carregar_dados(data)
    runs = dados["run"]
    is_primary = dados["IsPrimary"]

    run_ids = np.unique(runs)
    prim_counts = np.array([int(np.sum((runs == r) & (is_primary == 1))) for r in run_ids])
    sec_counts = np.array([int(np.sum((runs == r) & (is_primary == 0))) for r in run_ids])

    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    x = np.arange(len(run_ids))
    width = 0.38

    bars_p = plt.bar(x - width / 2, prim_counts, width,
                     color=COLOR_PRIMARY, edgecolor="black", linewidth=0.6,
                     label="Primários (IsPrimary = 1)")
    bars_s = plt.bar(x + width / 2, sec_counts, width,
                     color=COLOR_SECONDARY, edgecolor="black", linewidth=0.6,
                     label="Secundários (IsPrimary = 0)")

    plt.yscale("log")
    plt.xticks(x, [f"Run {int(r)}" for r in run_ids])
    plt.xlabel("Run", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Número de vértices (escala log)", fontsize=12, fontweight="bold", labelpad=10)
    plt.title("Vértices hadrónicos primários vs secundários por run",
              fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)

    # anotar a contagem exacta no topo de cada barra
    for bars in (bars_p, bars_s):
        for bar in bars:
            altura = bar.get_height()
            if altura > 0:
                plt.annotate(
                    f"{int(altura):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, altura),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5,
                )

    plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=10, loc="upper right")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "vertex_contagem_por_run.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {out_path}")

    # resumo no terminal
    print("  Contagem por run (primários / secundários):")
    for r, p, s in zip(run_ids, prim_counts, sec_counts):
        print(f"    Run {int(r)}: {p:,} primários | {s:,} secundários")


def main():
    """Carrega os dados uma única vez e gera as três figuras de análise."""
    print("\nAnálise de vértices hadrónicos (TTree 'hadronicVertex')")
    dados = _carregar_dados()
    plot_vertex_z_overlay(dados)
    plot_vertex_z_individual(dados)
    plot_contagem_por_run(dados)
    print(f"[OK] hadronic_vertex concluído — gráficos em {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
