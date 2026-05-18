"""
momentum_distribution.py
------------------------
Análise da distribuição do momento das partículas.

Análises incluídas:
    - Distribuição do momento na componente Z para muões e piões.
    - Distribuição do momento na componente Z para piões primários e secundários.
"""

import os
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot


def load_tracks_data(data):
    """
    Função auxiliar para carregar os dados de trajetórias a partir de um ficheiro,
    uma lista de ficheiros, ou utilizar diretamente um dicionário de dados já carregado.
    """
    if isinstance(data, dict):
        return data

    file_paths = []
    if isinstance(data, str):
        if os.path.isdir(data):
            file_paths = glob.glob(os.path.join(data, "AmberTarget_Run_*.root"))
        elif os.path.isfile(data):
            file_paths = [data]
    elif isinstance(data, (list, tuple)):
        file_paths = list(data)

    # Se não foi fornecido um caminho válido, tenta procurar nos locais habituais
    if not file_paths:
        search_dirs = ["ROOTFILES", "data", "."]
        for s_dir in search_dirs:
            matches = glob.glob(os.path.join(s_dir, "AmberTarget_Run_*.root"))
            if matches:
                file_paths = sorted(matches)
                break

    if not file_paths:
        raise FileNotFoundError(
            "Nenhum ficheiro ROOT de simulação (AmberTarget_Run_*.root) foi encontrado nas pastas habituais."
        )

    print(f"A carregar dados de: {file_paths}")

    # Concatenar dados de todas as runs para maior precisão e melhor estatística
    branches = ["particlePDG", "pZ_GeV", "IsPrimary"]
    combined_data = uproot.concatenate(
        [f"{path}:tracksData" for path in file_paths],
        expressions=branches,
        library="np"
    )
    return combined_data


def plot_momentum_muons_pions(data):
    """
    Distribuição do momento na componente Z para muões e piões.
    Considera apenas valores positivos de pZ, uma vez que a escala logarítmica
    não representa valores nulos ou negativos. Guarda o gráfico em plots/momentum/.
    """
    tracks = load_tracks_data(data)
    pdg = tracks["particlePDG"]
    pz = tracks["pZ_GeV"]

    # Máscaras de seleção de partículas
    muons_mask = np.isin(pdg, [13, -13])      # mu- (13) e mu+ (-13)
    pions_mask = np.isin(pdg, [211, -211])    # pi+ (211) e pi- (-211)

    positive_pz = pz > 0
    muons_pz = pz[muons_mask & positive_pz]
    pions_pz = pz[pions_mask & positive_pz]

    # Estilo profissional
    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    # Definir limites e bins adequados
    # Muões e piões têm distribuições de momento muito distintas (piões vão até ~200 GeV)
    bins = np.logspace(np.log10(1e-2), np.log10(200), 80)

    # Cores modernas e premium
    color_muons = "#1f77b4"  # Azul Cobalto Elegante
    color_pions = "#ff7f0e"  # Coral / Laranja Vibrante

    plt.hist(
        muons_pz,
        bins=bins,
        histtype="stepfilled",
        alpha=0.7,
        label=r"Muões ($\mu^{\pm}$)",
        color=color_muons,
        edgecolor=color_muons,
        linewidth=1.5
    )

    plt.hist(
        pions_pz,
        bins=bins,
        histtype="stepfilled",
        alpha=0.55,
        label=r"Piões ($\pi^{\pm}$)",
        color=color_pions,
        edgecolor=color_pions,
        linewidth=1.5
    )

    # Configuração dos eixos
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Momento Longitudinal $p_Z$ [GeV/c]", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Número de Trajetórias (Escala Log)", fontsize=12, fontweight="bold", labelpad=10)
    plt.title("Distribuição do Momento Longitudinal ($p_Z$) nos Detetores", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Legenda e layout
    plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=11, loc="upper right")
    plt.tight_layout()

    # Criar pasta plots/momentum/ se não existir
    os.makedirs(os.path.join("plots", "momentum"), exist_ok=True)
    out_path = os.path.join("plots", "momentum", "momentum_muons_pions.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {out_path}")


def plot_momentum_primary_secondary_pions(data):
    """
    Distribuição do momento na componente Z para piões primários e secundários.
    Considera apenas valores positivos de pZ, uma vez que a escala logarítmica
    não representa valores nulos ou negativos. Guarda o gráfico em plots/momentum/.
    """
    tracks = load_tracks_data(data)
    pdg = tracks["particlePDG"]
    pz = tracks["pZ_GeV"]
    is_primary = tracks["IsPrimary"]

    # Filtrar piões
    pions_mask = np.isin(pdg, [211, -211]) & (pz > 0)
    pions_pz = pz[pions_mask]
    pions_pri = is_primary[pions_mask]

    # Separar em primários e secundários
    primary_pions_pz = pions_pz[pions_pri == 1]
    secondary_pions_pz = pions_pz[pions_pri == 0]

    # Estilo profissional
    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    # Bins logarítmicos
    bins = np.logspace(np.log10(1e-2), np.log10(200), 80)

    # Cores modernas e premium
    color_primary = "#2ca02c"    # Verde Esmeralda
    color_secondary = "#d62728"  # Vermelho Carmim

    plt.hist(
        primary_pions_pz,
        bins=bins,
        histtype="stepfilled",
        alpha=0.7,
        label=r"Piões Primários ($\pi^{\pm}$)",
        color=color_primary,
        edgecolor=color_primary,
        linewidth=1.5
    )

    plt.hist(
        secondary_pions_pz,
        bins=bins,
        histtype="stepfilled",
        alpha=0.5,
        label=r"Piões Secundários ($\pi^{\pm}$)",
        color=color_secondary,
        edgecolor=color_secondary,
        linewidth=1.5
    )

    # Configuração dos eixos
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Momento Longitudinal $p_Z$ [GeV/c]", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Número de Trajetórias (Escala Log)", fontsize=12, fontweight="bold", labelpad=10)
    plt.title("Comparação de Momento: Piões Primários vs. Secundários", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Legenda e layout
    plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", fontsize=11, loc="upper right")
    plt.tight_layout()

    out_path = os.path.join("plots", "momentum", "momentum_pions_generation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado em: {out_path}")
