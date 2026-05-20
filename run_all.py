"""
run_all.py — ponto de entrada único para toda a análise
---------------------------------------------------------
Executa, pela ordem correcta, todos os scripts de análise implementados.
Os scripts dos colegas (hit_distribution, hadronic_vertex, etc.) são chamados
automaticamente quando estiverem implementados.

Uso:
    python run_all.py              # executa tudo
    python run_all.py --energy     # só deposição de energia
    python run_all.py --ml         # só ML (requer energy já executado)
"""

import sys
import os
import argparse
import importlib
import traceback

# garantir que os imports relativos funcionam a partir de qualquer directório
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# Utilitário
# ---------------------------------------------------------------------------

def run_step(label, module_name, main_fn="main"):
    """Importa um módulo de análise e chama a sua função principal."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        mod = importlib.import_module(f"analysis.{module_name}")
        if hasattr(mod, main_fn):
            getattr(mod, main_fn)()
        else:
            print(f"  [INFO] {module_name}.py não tem função '{main_fn}()' — a saltar.")
    except NotImplementedError:
        print(f"  [INFO] {module_name}.py ainda não implementado — a saltar.")
    except Exception as exc:
        print(f"  [ERRO] {module_name}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Executar todas as análises do AMBER Target."
    )
    parser.add_argument("--energy",  action="store_true",
                        help="Só deposição de energia + ML")
    parser.add_argument("--ml",      action="store_true",
                        help="Só classificador ML")
    args = parser.parse_args()

    run_energy = not any([args.energy, args.ml]) or args.energy
    run_ml     = not any([args.energy, args.ml]) or args.ml
    run_others = not any([args.energy, args.ml])

    print("\n" + "=" * 60)
    print("  AMBER Target — Pipeline completo de análise")
    print("  Universidade de Aveiro, TECII 2024/2025")
    print("=" * 60)

    # 1. Deposição de energia (Diana)
    if run_energy:
        # energy_deposition.py usa __name__ == "__main__" — invocar directamente
        print(f"\n{'='*60}")
        print("  [1/6] Deposição de energia (energy_deposition.py)")
        print(f"{'='*60}")
        try:
            import runpy
            runpy.run_path(
                os.path.join(ROOT_DIR, "analysis", "energy_deposition.py"),
                run_name="__main__",
            )
        except Exception as exc:
            print(f"  [ERRO] energy_deposition.py: {exc}")
            traceback.print_exc()

    # 2. Classificador ML (Diana)
    if run_ml:
        print(f"\n{'='*60}")
        print("  [2/6] Identificação de partículas por ML (particle_id_ml.py)")
        print(f"{'='*60}")
        try:
            import runpy
            runpy.run_path(
                os.path.join(ROOT_DIR, "analysis", "particle_id_ml.py"),
                run_name="__main__",
            )
        except Exception as exc:
            print(f"  [ERRO] particle_id_ml.py: {exc}")
            traceback.print_exc()

    # 3–6. Scripts dos colegas (executados quando implementados)
    if run_others:
        run_step("[3/6] Distribuição de hits",
                 "hit_distribution")
        run_step("[4/6] Vértices hadrónicos",
                 "hadronic_vertex")
        run_step("[5/6] Distribuição temporal",
                 "temporal_distribution")
        run_step("[6/6] Distribuição do momento",
                 "momentum_distribution")

    print("\n" + "=" * 60)
    print("  Pipeline concluído. Gráficos em: plots/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
