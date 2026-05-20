#!/usr/bin/env python
"""
run_momentum_analysis.py
------------------------
Script para executar a análise de momento de José e gerar os gráficos.
"""

import os
import sys
import glob
from analysis.momentum_distribution import (
    plot_momentum_muons_pions,
    plot_momentum_primary_secondary_pions
)

def main():
    print("==================================================")
    print("   TECII — Análise da Distribuição de Momento")
    print("==================================================")
    
    # Pasta contendo os dados ROOT (verifica 'data' ou 'ROOTFILES')
    data_dir = None
    for candidate in ("data", "ROOTFILES"):
        if glob.glob(os.path.join(candidate, "AmberTarget_Run_*.root")):
            data_dir = candidate
            break

    if data_dir is None:
        data_dir = "data"
        
    print(f"Diretório de dados selecionado: '{data_dir}'")
    
    try:
        # Executar os plots
        print("\nGerando gráfico: Distribuição de Momento (Muões vs Piões)...")
        plot_momentum_muons_pions(data_dir)
        
        print("Gerando gráfico: Comparação de Piões (Primários vs Secundários)...")
        plot_momentum_primary_secondary_pions(data_dir)
        
        print("\n==================================================")
        print("Execução concluída com sucesso.")
        print("Os gráficos foram guardados em: plots/momentum/")
        print("Ficheiros gerados:")
        print("  - plots/momentum/momentum_muons_pions.png")
        print("  - plots/momentum/momentum_pions_generation.png")
        print("==================================================")
        
    except Exception as e:
        print(f"\nErro durante a execução: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
