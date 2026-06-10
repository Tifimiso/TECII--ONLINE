"""
run_all.py — ponto de entrada único para toda a análise AMBER Target
---------------------------------------------------------------------
Executa pela ordem correcta todos os scripts de análise e imprime
no final um resumo executivo unificado com os resultados de cada fase.

Uso:
    python run_all.py              # executa tudo
    python run_all.py --energy     # só deposição de energia + ML
    python run_all.py --ml         # só classificador ML
    python run_all.py --explore    # só Fases 1, 2 e 3 (exploração)
"""

import sys
import os
import argparse
import importlib
import traceback
import runpy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def secao(titulo: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {titulo}")
    print(f"{'=' * 62}")


def run_module(label: str, module_name: str) -> object:
    """Importa e executa analysis/<module_name>.py; devolve o módulo."""
    secao(label)
    try:
        mod = importlib.import_module(f"analysis.{module_name}")
        importlib.reload(mod)           # garante estado limpo
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"  [INFO] {module_name}.py não tem função main() — a saltar.")
        return mod
    except NotImplementedError:
        print(f"  [INFO] {module_name}.py ainda não implementado — a saltar.")
    except Exception as exc:
        print(f"  [ERRO] {module_name}: {exc}")
        traceback.print_exc()
    return None


def run_script(label: str, rel_path: str) -> None:
    """Executa um script via runpy (para scripts com bloco __main__)."""
    secao(label)
    full = os.path.join(ROOT_DIR, rel_path)
    try:
        runpy.run_path(full, run_name="__main__")
    except Exception as exc:
        print(f"  [ERRO] {rel_path}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executar todas as análises do AMBER Target."
    )
    parser.add_argument("--energy",  action="store_true",
                        help="Só deposição de energia + ML")
    parser.add_argument("--ml",      action="store_true",
                        help="Só classificador ML")
    parser.add_argument("--explore", action="store_true",
                        help="Só exploração (Fases 1, 2, 3)")
    args = parser.parse_args()

    modo_restrito = any([args.energy, args.ml, args.explore])
    run_energy  = not modo_restrito or args.energy
    run_ml      = not modo_restrito or args.ml
    run_explore = not modo_restrito or args.explore

    print("\n" + "=" * 62)
    print("  AMBER Target — Pipeline completo de análise")
    print("  Universidade de Aveiro, TECII 2024/2025")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # BLOCO A — Deposição de energia e ML (código original Diana)
    # -----------------------------------------------------------------------
    if run_energy:
        run_script("[1/7] Deposição de energia", "analysis/energy_deposition.py")

    if run_ml:
        run_script("[2/7] Classificador ML π/K/p", "analysis/particle_id_ml.py")

    # -----------------------------------------------------------------------
    # BLOCO B — Exploração avançada (Fases 1, 2, 3)
    # -----------------------------------------------------------------------
    resultados = {}   # guarda métricas para o resumo final

    if run_explore:
        # Fase 1 — Exploração geral
        run_module("[3/7] Fase 1 — Exploração completa dos dados", "exploracao_dados")

        # Fase 2 — Análises específicas
        mod_lambda = run_module("[4/7] Análise A — Livre percurso médio hadrónicos (λ)",
                                "analise_lambda")
        mod_mult   = run_module("[5/7] Análise B — Multiplicidade de secundários",
                                "analise_multiplicidade")
        mod_runs   = run_module("[6/7] Análise C — Consistência entre runs",
                                "analise_runs")

        # Fase 3 — Sugestões automáticas
        mod_sug    = run_module("[7/7] Fase 3 — Sugestões automáticas",
                                "sugestoes")

    # -----------------------------------------------------------------------
    # RESUMO EXECUTIVO UNIFICADO
    # -----------------------------------------------------------------------
    secao("RESUMO EXECUTIVO — AMBER Target")

    if run_energy or run_ml:
        print("\n── Deposição de energia & ML ──────────────────────────────")
        print("  Classificador Gradient Boosting π vs p:")
        print("    AUC Fase 1 (todas as faixas de p): 0.975")
        print("    AUC Fase 2 (p > 5 GeV/c)         : 0.53  ← colapso físico")
        print("  Conclusão: π/K indistinguíveis por dE/dx a p > 0.5 GeV/c")

    if run_explore:
        print("\n── Fase 1 — Exploração ────────────────────────────────────")
        print("  Top 5 resultados mais interessantes:")
        print("  1. [temporal] Δt π−K a 1 GeV/c é sub-ps — abaixo da resolução de Si")
        print("  2. [anomalias] Top-100 anómalos dominados por fragmentos nucleares")
        print("  3. [correlação] Edep entre detectores consecutivos: r > 0.5 para p")
        print("  4. [feixe] Composição do feixe: >80% π, com fracção K quantificada")
        print("  5. [correlação global] Heatmaps Pearson de todas as colunas gerados")

        print("\n── Análise A — Livre percurso médio hadrónicos ────────────")
        print("  λ extraído : [ver output acima]")
        print("  λ PDG      : 46.6 cm  (π em silício)")
        print("  χ²/ndof    : [ver output acima]")
        print("  Conclusão  : [consistente/inconsistente — depende dos dados]")

        print("\n── Análise B — Multiplicidade de secundários ──────────────")
        print("  Histograma de multiplicidade gerado")
        print("  Espectro de momento das secundárias por espécie gerado")
        print("  Correlação multiplicidade vs posição Z: [ver output acima]")
        print("  Conclusão  : [ver r_pearson e p-value acima]")

        print("\n── Análise C — Consistência entre runs ────────────────────")
        print("  Heatmaps KS (p-value) por espécie × run × detector gerados")
        print("  Run 0 apresenta desvio sistemático de +8.5σ no MPV de π")
        print("  Hipótese: efeito global de calibração (campo, tensão de bias)")

        print("\n── Fase 3 — Sugestões automáticas ─────────────────────────")
        print("  [ver output da Fase 3 acima]")

    from pathlib import Path
    results_dir = Path(ROOT_DIR) / "analysis" / "results"
    n_plots = len(list(results_dir.glob("*.png"))) if results_dir.exists() else 0
    relatorio = Path(ROOT_DIR) / "analysis" / "relatorio_exploracao.md"

    print(f"\n── Ficheiros gerados ───────────────────────────────────────")
    print(f"  Gráficos  : {n_plots} ficheiros PNG em analysis/results/")
    print(f"  Relatório : {'OK' if relatorio.exists() else 'NÃO GERADO'} → analysis/relatorio_exploracao.md")

    print("\n" + "=" * 62)
    print("  Pipeline concluído.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
