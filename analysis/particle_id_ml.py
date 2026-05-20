"""
particle_id_ml.py  (v2)
-----------------------
Classificador de Aprendizagem Automática para identificação de partículas (PID).

Melhorias em relação à versão 1:
  1. Classificação a 3 classes: pião (0), kaão (1), protão (2).
  2. Balanceamento de classes por sample_weight — corrige o baixo recall
     dos protões e kaões causado pelo desequilíbrio do dataset.
  3. Validação cruzada estratificada a 5 folds — confirma a estabilidade
     do modelo e evita sobre-ajuste ao split treino/teste.
  4. HistGradientBoostingClassifier: versão moderna e muito mais rápida do
     Gradient Boosting, adequada para datasets grandes (sklearn >= 0.21).

Filtros aplicados (iguais ao energy_deposition.py):
    - EdepDet0_keV > 10  (remover ruído)
    - momentum_GeV > 0.5 (regime relativista)
    - espécies: piões (|PDG|=211), kaões (|PDG|=321), protões (PDG=2212)

Uso:
    python analysis/particle_id_ml.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import uproot
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance

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

CLASSES      = [0, 1, 2]
CLASS_NAMES  = ["Pião", "Kaão", "Protão"]
CLASS_COLORS = ["red", "cyan", "magenta"]

# fundo suave para os 3 territórios de decisão
BG_CMAP = ListedColormap(["#FFD6D6", "#D6F5FF", "#FFD6FF"])

FEATURE_NAMES_PT = [
    "dE/dx Detetor 0 (keV)",
    "dE/dx Detetor 1 (keV)",
    "dE/dx Detetor 2 (keV)",
    "dE/dx Detetor 3 (keV)",
    "Momento (GeV/c)",
]

CLF_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# 1. Leitura e preparação dos dados
# ---------------------------------------------------------------------------

def carregar_dados():
    """Lê os 4 ficheiros ROOT com uproot — 3 classes: π, K, p."""
    branches = [
        "particlePDG", "EdepDet0_keV", "EdepDet1_keV",
        "EdepDet2_keV", "EdepDet3_keV", "momentum_GeV",
    ]
    arrays = []
    for fname in DATA_FILES:
        with uproot.open(fname) as f:
            arrays.append(f["tracksData"].arrays(branches, library="np"))

    data = {k: np.concatenate([a[k] for a in arrays]) for k in branches}

    pdg   = data["particlePDG"]
    edep0 = data["EdepDet0_keV"]
    mom   = data["momentum_GeV"]

    mask = (
        ((np.abs(pdg) == 211) | (np.abs(pdg) == 321) | (pdg == 2212)) &
        (edep0 > 10) &
        (mom > 0.5)
    )

    pdg_sel = pdg[mask]
    y = np.zeros(mask.sum(), dtype=int)
    y[np.abs(pdg_sel) == 321] = 1   # kaão
    y[pdg_sel == 2212]        = 2   # protão

    X = np.column_stack([
        data["EdepDet0_keV"][mask],
        data["EdepDet1_keV"][mask],
        data["EdepDet2_keV"][mask],
        data["EdepDet3_keV"][mask],
        data["momentum_GeV"][mask],
    ])

    print(f"  Total de tracks após filtros: {mask.sum():,}")
    for i, nome in enumerate(CLASS_NAMES):
        n = (y == i).sum()
        print(f"  {nome:8s}: {n:>8,}  ({100*n/len(y):.1f}%)")

    return X, y


# ---------------------------------------------------------------------------
# 2. Validação cruzada
# ---------------------------------------------------------------------------

def validacao_cruzada(X, y, n_splits=5):
    """5-fold stratified CV — reporta AUC e accuracy por fold."""
    print(f"\n  Validação cruzada ({n_splits} folds estratificados):")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)
    acc_folds = []
    auc_folds = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        sw = compute_sample_weight("balanced", y_tr)
        clf_cv = HistGradientBoostingClassifier(**CLF_PARAMS)
        clf_cv.fit(X_tr, y_tr, sample_weight=sw)

        y_pred = clf_cv.predict(X_val)
        y_prob = clf_cv.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        auc_macro = roc_auc_score(
            y_val, y_prob, multi_class="ovr", average="macro"
        )
        acc_folds.append(acc)
        auc_folds.append(auc_macro)
        print(f"    Fold {fold}: accuracy = {acc:.4f}  |  AUC macro = {auc_macro:.4f}")

    print(f"  CV accuracy : {np.mean(acc_folds):.4f} ± {np.std(acc_folds):.4f}")
    print(f"  CV AUC macro: {np.mean(auc_folds):.4f} ± {np.std(auc_folds):.4f}")
    return acc_folds, auc_folds


# ---------------------------------------------------------------------------
# 3. Treino do modelo final
# ---------------------------------------------------------------------------

def treinar(X_train, y_train):
    sw = compute_sample_weight("balanced", y_train)
    clf = HistGradientBoostingClassifier(**CLF_PARAMS)
    clf.fit(X_train, y_train, sample_weight=sw)
    return clf


# ---------------------------------------------------------------------------
# 4. Gráficos
# ---------------------------------------------------------------------------

def plot_roc_e_confusion(clf, X_test, y_test, acc_folds, auc_folds):
    """Curvas ROC (OvR, 3 classes) + matriz de confusão 3×3."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    y_bin  = label_binarize(y_test, classes=CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- ROC ---
    ax = axes[0]
    auc_vals = []
    for i, (nome, cor) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_vals.append(roc_auc)
        ax.plot(fpr, tpr, color=cor, lw=2,
                label=f"{nome}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos", fontsize=12)
    ax.set_ylabel("Taxa de verdadeiros positivos", fontsize=12)
    ax.set_title("Curvas ROC — uma vs. resto (OvR)\n3 classes: π, K, p", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # caixa com resultados de CV
    cv_txt = (
        f"Validação cruzada (5-fold)\n"
        f"AUC  = {np.mean(auc_folds):.4f} ± {np.std(auc_folds):.4f}\n"
        f"Acc  = {np.mean(acc_folds):.4f} ± {np.std(acc_folds):.4f}"
    )
    ax.text(0.38, 0.08, cv_txt, transform=ax.transAxes,
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))

    # --- Confusion matrix ---
    ax2 = axes[1]
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    disp.plot(ax=ax2, colorbar=True, cmap="Blues", values_format=".1%")
    ax2.set_title("Matriz de confusão (normalizada)\n3 classes", fontsize=12)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/ml_roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    return auc_vals


def plot_feature_importance(clf, X_test, y_test):
    """
    Importância das features por permutation importance:
    mede quanto a accuracy cai quando cada feature é permutada aleatoriamente.
    Mais robusto e interpretável que a importância baseada em ganho de Gini.
    Usa uma amostra de 3000 eventos do conjunto de teste para velocidade.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_sample = min(3000, len(y_test))
    rng = np.random.default_rng(RANDOM_STATE)
    idx_sample = rng.choice(len(y_test), size=n_sample, replace=False)

    result = permutation_importance(
        clf, X_test[idx_sample], y_test[idx_sample],
        n_repeats=15, random_state=RANDOM_STATE,
        scoring="balanced_accuracy",
    )
    importances = result.importances_mean
    importances_std = result.importances_std

    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        [FEATURE_NAMES_PT[i] for i in idx],
        importances[idx],
        xerr=importances_std[idx],
        color="steelblue", edgecolor="white",
        capsize=4, error_kw={"ecolor": "gray", "lw": 1.5},
    )
    ax.set_xlabel("Queda em balanced accuracy\n(permutation importance)", fontsize=11)
    ax.set_title("Importância das variáveis (permutation importance)\n"
                 "3 classes: pião, kaão, protão — 15 repetições", fontsize=12)
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/ml_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_decision_boundary(clf, X, y):
    """Fronteira de decisão 2D — dE/dx (det 0, MeV) vs momento — 3 classes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KEV_TO_MEV = 1e-3
    rng = np.random.default_rng(RANDOM_STATE)

    # amostra estratificada de 5000 eventos
    n_sample = 5000
    indices = []
    for cls in CLASSES:
        idx_cls = np.where(y == cls)[0]
        n_cls = min(int(n_sample * len(idx_cls) / len(y)), len(idx_cls))
        indices.append(rng.choice(idx_cls, size=n_cls, replace=False))
    sel = np.concatenate(indices)

    X_s = X[sel]
    y_s = y[sel]

    edep_mev = X_s[:, 0] * KEV_TO_MEV
    mom_gev  = X_s[:, 4]

    # grelha de decisão
    edep_min = max(edep_mev.min() * 0.9,  0.001)
    edep_max = edep_mev.max() * 1.1
    mom_min  = mom_gev.min()  * 0.95
    mom_max  = mom_gev.max()  * 1.05

    n_grid = 250
    xx, yy = np.meshgrid(
        np.linspace(edep_min, edep_max, n_grid),
        np.linspace(mom_min,  mom_max,  n_grid),
    )

    med1 = np.median(X[:, 1])
    med2 = np.median(X[:, 2])
    med3 = np.median(X[:, 3])

    grid_features = np.column_stack([
        xx.ravel() / KEV_TO_MEV,
        np.full(xx.size, med1),
        np.full(xx.size, med2),
        np.full(xx.size, med3),
        yy.ravel(),
    ])

    Z = clf.predict(grid_features).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                cmap=BG_CMAP, alpha=0.55)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors="black",
               linewidths=1.2, linestyles="--")

    s_map  = {0: "red", 1: "darkcyan", 2: "magenta"}
    labels = {0: "Pião", 1: "Kaão", 2: "Protão"}
    for cls in CLASSES:
        m = y_s == cls
        ax.scatter(edep_mev[m], mom_gev[m],
                   c=s_map[cls], s=5, alpha=0.4,
                   label=f"{labels[cls]} (n={m.sum()})")

    patches = [
        mpatches.Patch(color="#FFD6D6", label="Região: Pião"),
        mpatches.Patch(color="#D6F5FF", label="Região: Kaão"),
        mpatches.Patch(color="#FFD6FF", label="Região: Protão"),
    ]
    leg1 = ax.legend(handles=patches, fontsize=9, loc="upper right",
                     title="Regiões de decisão")
    ax.add_artist(leg1)
    ax.legend(fontsize=9, loc="upper left", markerscale=3,
              title="Eventos (amostra)")

    ax.set_xlabel("dE/dx — Detetor 0 (MeV)", fontsize=12)
    ax.set_ylabel("Momento (GeV/c)", fontsize=12)
    ax.set_title("Fronteira de decisão — 3 classes (π, K, p)\n"
                 "amostra de 5 000 eventos; linhas tracejadas = fronteiras",
                 fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/ml_decision_boundary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_cv_scores(acc_folds, auc_folds):
    """Gráfico de barras com os scores de cada fold de CV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    folds = np.arange(1, len(acc_folds) + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(folds - width/2, acc_folds, width,
                label="Accuracy", color="steelblue", alpha=0.85)
    b2 = ax.bar(folds + width/2, auc_folds, width,
                label="AUC macro (OvR)", color="darkorange", alpha=0.85)

    ax.bar_label(b1, fmt="%.4f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.4f", fontsize=8, padding=2)

    ax.axhline(np.mean(acc_folds), color="steelblue",
               linestyle="--", lw=1.5,
               label=f"Média accuracy = {np.mean(acc_folds):.4f}")
    ax.axhline(np.mean(auc_folds), color="darkorange",
               linestyle="--", lw=1.5,
               label=f"Média AUC = {np.mean(auc_folds):.4f}")

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validação cruzada — 5 folds estratificados\n"
                 "HistGradient Boosting, 3 classes (π, K, p)", fontsize=12)
    ax.set_xticks(folds)
    ax.set_ylim(0.70, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/ml_cv_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# 5. Programa principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Classificador ML para PID — 3 classes (π, K, p)  v2")
    print("=" * 60)

    # --- leitura ---
    print("\n[1/5] A carregar dados...")
    X, y = carregar_dados()

    # --- split ---
    print("\n[2/5] A dividir conjuntos treino/teste (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Treino: {len(y_train):,}  |  Teste: {len(y_test):,}")

    # --- validação cruzada ---
    print("\n[3/5] A efectuar validação cruzada (5 folds)...")
    acc_folds, auc_folds = validacao_cruzada(X, y, n_splits=5)

    # --- treino final ---
    print("\n[4/5] A treinar modelo final (com balanceamento de classes)...")
    clf = treinar(X_train, y_train)
    print("  Treino concluído.")

    # --- avaliação ---
    print("\n[5/5] A avaliar e gerar gráficos...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    y_bin  = label_binarize(y_test, classes=CLASSES)

    acc      = accuracy_score(y_test, y_pred)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    auc_macro = roc_auc_score(y_test, y_prob,
                              multi_class="ovr", average="macro")

    print("\n" + "=" * 60)
    print("RESULTADOS (conjunto de teste)")
    print("=" * 60)
    print(f"  AUC macro (OvR)       : {auc_macro:.4f}")
    print(f"  Balanced accuracy     : {bal_acc:.4f}  ({bal_acc*100:.2f}%)")
    print(f"  Accuracy (não balanceada): {acc:.4f}  ({acc*100:.2f}%)")
    print("  Nota: accuracy não balanceada é enganosa com 3 classes desequilibradas")
    print(f"  Baseline aleatório (3 classes): 33.3%")
    print("\nRelatório de classificação:")
    print(classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    ))

    print("A gerar gráficos...")
    auc_por_classe = plot_roc_e_confusion(clf, X_test, y_test,
                                          acc_folds, auc_folds)
    plot_feature_importance(clf, X_test, y_test)
    plot_decision_boundary(clf, X, y)
    plot_cv_scores(acc_folds, auc_folds)

    print(f"\nAUC por classe:")
    for nome, val in zip(CLASS_NAMES, auc_por_classe):
        print(f"  {nome:8s}: {val:.4f}")

    print("\nDone. Gráficos guardados em:", OUTPUT_DIR)
