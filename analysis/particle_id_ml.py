"""
particle_id_ml.py
-----------------
Classificador de Aprendizagem Automática para identificação de partículas (PID).

Distingue piões de protões usando a deposição de energia nos 4 detetores e o
momento como variáveis de entrada, com um GradientBoostingClassifier (sklearn).

Filtros aplicados (iguais aos de energy_deposition.py):
    - partícula é pião (abs(PDG)==211) ou protão (PDG==2212)
    - EdepDet0_keV > 10  (remover ruído)
    - momentum_GeV > 0.5 (regime relativista)

Uso:
    python analysis/particle_id_ml.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

import uproot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, accuracy_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DATA_FILES = [
    "data/AmberTarget_Run_0.root",
    "data/AmberTarget_Run_1.root",
    "data/AmberTarget_Run_2.root",
    "data/AmberTarget_Run_3.root",
]
OUTPUT_DIR = "plots/energy"
RANDOM_STATE = 42

FEATURE_NAMES_PT = [
    "dE/dx Detetor 0 (keV)",
    "dE/dx Detetor 1 (keV)",
    "dE/dx Detetor 2 (keV)",
    "dE/dx Detetor 3 (keV)",
    "Momento (GeV/c)",
]

# ---------------------------------------------------------------------------
# 1. Leitura e filtragem dos dados
# ---------------------------------------------------------------------------

def carregar_dados():
    """Lê os 4 ficheiros ROOT com uproot e concatena num único array numpy."""
    branches = [
        "particlePDG", "EdepDet0_keV", "EdepDet1_keV",
        "EdepDet2_keV", "EdepDet3_keV", "momentum_GeV",
    ]

    arrays = []
    for fname in DATA_FILES:
        with uproot.open(fname) as f:
            tree = f["tracksData"]
            arrays.append(tree.arrays(branches, library="np"))

    data = {k: np.concatenate([a[k] for a in arrays]) for k in branches}

    pdg = data["particlePDG"]
    edep0 = data["EdepDet0_keV"]
    mom = data["momentum_GeV"]

    # filtros: apenas piões e protões, com deposição significativa e momento relativista
    mask = (
        ((np.abs(pdg) == 211) | (pdg == 2212)) &
        (edep0 > 10) &
        (mom > 0.5)
    )

    print(f"  Total de tracks após filtros: {mask.sum():,}")
    print(f"  Piões:  {((np.abs(pdg[mask]) == 211)).sum():,}")
    print(f"  Protões: {(pdg[mask] == 2212).sum():,}")

    X = np.column_stack([
        data["EdepDet0_keV"][mask],
        data["EdepDet1_keV"][mask],
        data["EdepDet2_keV"][mask],
        data["EdepDet3_keV"][mask],
        data["momentum_GeV"][mask],
    ])
    # y = 0 → pião, y = 1 → protão
    y = (data["particlePDG"][mask] == 2212).astype(int)

    return X, y


# ---------------------------------------------------------------------------
# 2. Treino do classificador
# ---------------------------------------------------------------------------

def treinar_classificador(X_train, y_train):
    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# 3. Gráficos
# ---------------------------------------------------------------------------

def plot_roc_e_confusion(clf, X_test, y_test):
    """Curva ROC + matriz de confusão numa figura com dois subplots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- subplot esquerdo: ROC ---
    ax = axes[0]
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Classificador aleatório")
    ax.set_xlabel("Taxa de falsos positivos", fontsize=12)
    ax.set_ylabel("Taxa de verdadeiros positivos", fontsize=12)
    ax.set_title("Curva ROC — Gradient Boosting\n"
                 r"(pião vs protão)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- subplot direito: matriz de confusão ---
    ax2 = axes[1]
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Pião", "Protão"],
    )
    disp.plot(ax=ax2, colorbar=True, cmap="Blues",
              values_format=".1%")
    ax2.set_title("Matriz de confusão (normalizada)", fontsize=12)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/ml_roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    return roc_auc


def plot_feature_importance(clf):
    """Bar chart horizontal das importâncias das features."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    importances = clf.feature_importances_
    idx = np.argsort(importances)  # ordem crescente para o gráfico

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        [FEATURE_NAMES_PT[i] for i in idx],
        importances[idx],
        color="steelblue", edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("Importância relativa", fontsize=12)
    ax.set_title("Importância das variáveis — Gradient Boosting\n"
                 r"(pião vs protão)", fontsize=12)
    ax.set_xlim(0, importances.max() * 1.15)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/ml_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_decision_boundary(clf, X, y):
    """
    Gráfico 2D de dE/dx (detetor 0, em MeV) vs momento, com região de decisão
    do classificador como fundo colorido.
    Usa uma amostra de 5000 eventos para não tornar o gráfico ilegível.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3
    rng = np.random.default_rng(RANDOM_STATE)

    # amostra de 5000 eventos (estratificada por espécie)
    n_sample = min(5000, len(y))
    idx_pi = np.where(y == 0)[0]
    idx_pr = np.where(y == 1)[0]
    n_pi = min(int(n_sample * len(idx_pi) / len(y)), len(idx_pi))
    n_pr = min(n_sample - n_pi, len(idx_pr))

    sel_pi = rng.choice(idx_pi, size=n_pi, replace=False)
    sel_pr = rng.choice(idx_pr, size=n_pr, replace=False)
    sel = np.concatenate([sel_pi, sel_pr])

    X_s = X[sel]
    y_s = y[sel]

    edep_mev = X_s[:, 0] * KEV_TO_MEV   # detetor 0 em MeV
    mom_gev  = X_s[:, 4]                  # momento em GeV/c

    # grelha para a região de decisão
    edep_min, edep_max = edep_mev.min() * 0.9, edep_mev.max() * 1.1
    mom_min,  mom_max  = mom_gev.min()  * 0.95, mom_gev.max()  * 1.05

    n_grid = 200
    xx, yy = np.meshgrid(
        np.linspace(edep_min, edep_max, n_grid),
        np.linspace(mom_min,  mom_max,  n_grid),
    )

    # reconstruir features completas para a grelha
    # usar medianas das outras features para o fundo
    med_edep1 = np.median(X[:, 1])
    med_edep2 = np.median(X[:, 2])
    med_edep3 = np.median(X[:, 3])

    grid_edep_kev = xx.ravel() / KEV_TO_MEV
    grid_mom      = yy.ravel()
    grid_features = np.column_stack([
        grid_edep_kev,
        np.full_like(grid_edep_kev, med_edep1),
        np.full_like(grid_edep_kev, med_edep2),
        np.full_like(grid_edep_kev, med_edep3),
        grid_mom,
    ])

    Z = clf.predict_proba(grid_features)[:, 1].reshape(xx.shape)

    # paleta: azul-claro para pião (0), magenta-claro para protão (1)
    cmap_bg = mcolors.LinearSegmentedColormap.from_list(
        "pid_bg",
        [(0.85, 0.90, 1.0), (1.0, 0.85, 1.0)],  # azul claro → magenta claro
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(xx, yy, Z, levels=50, cmap=cmap_bg, alpha=0.75,
                     vmin=0.0, vmax=1.0)
    plt.colorbar(cf, ax=ax, label="Probabilidade (protão)")

    # linha de fronteira de decisão (p=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5,
               linestyles="--")

    # pontos
    mask_pi = y_s == 0
    mask_pr = y_s == 1
    ax.scatter(edep_mev[mask_pi], mom_gev[mask_pi],
               c="red", s=4, alpha=0.4, label=f"Pião (n={mask_pi.sum()})")
    ax.scatter(edep_mev[mask_pr], mom_gev[mask_pr],
               c="magenta", s=4, alpha=0.4, label=f"Protão (n={mask_pr.sum()})")

    ax.set_xlabel("dE/dx — Detetor 0 (MeV)", fontsize=12)
    ax.set_ylabel("Momento (GeV/c)", fontsize=12)
    ax.set_title("Fronteira de decisão — Gradient Boosting\n"
                 r"(pião vs protão, amostra de 5000 eventos)", fontsize=12)
    ax.legend(fontsize=10, markerscale=3)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/ml_decision_boundary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# 4. Programa principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Classificador ML para PID (pião vs protão)")
    print("=" * 60)

    # --- leitura ---
    print("\n[1/4] A carregar dados...")
    X, y = carregar_dados()

    # --- split ---
    print("\n[2/4] A dividir conjuntos treino/teste (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Treino: {len(y_train):,} eventos  |  Teste: {len(y_test):,} eventos")

    # --- treino ---
    print("\n[3/4] A treinar GradientBoostingClassifier...")
    clf = treinar_classificador(X_train, y_train)
    print("  Treino concluído.")

    # --- avaliação ---
    print("\n[4/4] A avaliar e gerar gráficos...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"  AUC score : {roc_auc:.4f}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print("\nRelatório de classificação:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Pião", "Protão"],
        digits=4,
    ))

    # --- gráficos ---
    print("A gerar gráficos...")
    plot_roc_e_confusion(clf, X_test, y_test)
    plot_feature_importance(clf)
    plot_decision_boundary(clf, X, y)

    print("\nDone. Gráficos guardados em:", OUTPUT_DIR)
