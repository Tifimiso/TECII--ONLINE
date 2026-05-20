"""
particle_id_ml.py  (v3 — classificador de produção)
----------------------------------------------------
Análise em duas fases, desenvolvida seguindo os princípios dos sistemas de
PID por aprendizagem automática em operação no CERN:
  - LHCb ProbNN: Gradient Boosted Decision Trees combinando 4 sub-detectores,
    em produção desde 2011 (JINST 14, 2019, P04006).
  - ALICE TPC: redes neuronais para PID por dE/dx no gás (CERN/LHCC 2000-001).
  - TMVA (ROOT Toolkit for Multivariate Analysis): framework HEP padrão para
    classificadores BDT/NN, conceptualmente idêntico ao sklearn.

FASE 1 — Análise exploratória (π, K, p, p > 0.5 GeV/c)
  Demonstra quantitativamente que K/π é fisicamente impossível por dE/dx a
  p > 0.5 GeV/c (precision dos kaões: 10.4%, AUC = 0.73).
  Justifica a restrição do domínio do classificador de produção.
  Gera: ml_roc_curve.png, ml_cv_scores.png, ml_decision_boundary.png,
        ml_feature_importance.png, ml_efficiency_purity.png,
        ml_auc_vs_momentum.png

FASE 2 — Classificador de produção (π vs p, p ∈ [0.5, 2.0] GeV/c)
  Domínio restrito à janela cinemática onde dE/dx tem poder discriminativo
  real (AUC > 0.93 por bin de momento). Inclui verificação de domínio de
  validade — eventos fora do intervalo são sinalizados como "não classificável".
  Limitações documentadas:
    - p ∈ [0.5, 1.0] GeV/c: protões com βγ ∈ [0.53, 1.07] (não plenamente
      relativistas; distribuição de Landau é uma aproximação).
    - Sem correcção de ângulo de incidência (dados não contêm direcção da track).
    - Treinado em simulação Geant4; desempenho em dados reais requereria
      re-treino com amostras de controlo cinemáticas.
  Gera: ml_binary_roc.png, ml_binary_cv.png, ml_binary_effpur.png,
        ml_binary_features.png, ml_binary_boundary.png

Uso:
    python analysis/particle_id_ml.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
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
# 5. Curva eficiência-pureza (π vs p)
# ---------------------------------------------------------------------------

def plot_efficiency_purity(clf, X_test, y_test):
    """
    Curva eficiência-pureza (precision-recall) para a separação π vs p.

    Num experimento real, o físico escolhe o ponto de operação em função do
    objectivo da análise:
      - Alta pureza (poucas contaminações) → menos eventos, resultado mais limpo
      - Alta eficiência (poucos eventos perdidos) → mais contaminação

    Usa os eventos π e p do conjunto de teste, com P(protão) como score.
    Normaliza para o caso binário: P(p | π ou p) = P(p) / (P(π) + P(p)).
    Mostra três pontos de operação representativos (90%, 95%, 99% pureza).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # filtrar apenas π (y=0) e p (y=2) do conjunto de teste
    mask_pip = (y_test == 0) | (y_test == 2)
    X_pp   = X_test[mask_pip]
    y_pp   = (y_test[mask_pip] == 2).astype(int)   # 0=pião, 1=protão

    y_prob_pp = clf.predict_proba(X_pp)
    # probabilidade normalizada para o caso binário π/p
    prob_pi = y_prob_pp[:, 0]
    prob_p  = y_prob_pp[:, 2]
    score   = prob_p / (prob_pi + prob_p + 1e-12)

    prec, rec, thresholds = precision_recall_curve(y_pp, score)
    ap = average_precision_score(y_pp, score)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- curva pureza vs eficiência ---
    ax = axes[0]
    ax.plot(rec, prec, color="steelblue", lw=2,
            label=f"AP = {ap:.4f}")
    ax.set_xlabel("Eficiência (recall de protões)", fontsize=12)
    ax.set_ylabel("Pureza (precision de protões)", fontsize=12)
    ax.set_title("Curva eficiência–pureza\nprotão vs pião", fontsize=12)
    ax.grid(True, alpha=0.3)

    # pontos de operação a 90%, 95%, 99% de pureza
    op_purities = [0.90, 0.95, 0.99]
    op_colors   = ["green", "orange", "red"]
    for target_purity, col in zip(op_purities, op_colors):
        # encontrar o threshold com pureza >= target
        valid = np.where(prec >= target_purity)[0]
        if len(valid) == 0:
            continue
        idx = valid[np.argmax(rec[valid])]   # maior eficiência a essa pureza
        ax.scatter(rec[idx], prec[idx], s=80, color=col, zorder=5,
                   label=f"Pureza {int(target_purity*100)}%: eff={rec[idx]:.2f}")

    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.3, 1.02)

    # --- eficiência e pureza vs threshold ---
    ax2 = axes[1]
    t_plot = thresholds
    ax2.plot(t_plot, prec[:-1], color="darkorange", lw=2, label="Pureza")
    ax2.plot(t_plot, rec[:-1],  color="steelblue",  lw=2, label="Eficiência")
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    ax2.plot(t_plot, f1, color="green", lw=2, linestyle="--", label="F1-score")
    ax2.axvline(0.5, color="gray", lw=1, linestyle=":", label="Threshold = 0.5")
    ax2.set_xlabel("Threshold de decisão P(protão)", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Pureza e eficiência vs threshold\nprotão vs pião", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/ml_efficiency_purity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")

    # imprimir pontos de operação
    print("  Pontos de operação π vs p:")
    for target_purity, col in zip(op_purities, op_colors):
        valid = np.where(prec >= target_purity)[0]
        if len(valid) == 0:
            print(f"    Pureza {int(target_purity*100)}%: não atingível")
            continue
        idx = valid[np.argmax(rec[valid])]
        thr = thresholds[idx] if idx < len(thresholds) else 1.0
        print(f"    Pureza {int(target_purity*100)}%: "
              f"eficiência = {rec[idx]:.3f}, threshold = {thr:.3f}")


# ---------------------------------------------------------------------------
# 6. Classificador por bin de momento (π vs p, só dE/dx)
# ---------------------------------------------------------------------------

def plot_auc_vs_momentum(X, y):
    """
    Treina classificadores binários π vs p em bins de momento,
    usando APENAS a deposição de energia nos 4 detetores como features
    (o momento é o que define o bin — simula o cenário real onde o
    espectrômetro magnético já fornece p e o dE/dx faz o PID).

    Resultado: AUC decresce com o momento, confirmando que a separação
    por dE/dx só é eficaz a baixo momento — resultado fisicamente esperado
    pela curva de Bethe-Bloch.

    Bins de momento: [0.5–1.0], [1.0–2.0], [2.0–5.0], [>5.0] GeV/c.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # apenas piões (y=0) e protões (y=2)
    mask_pp = (y == 0) | (y == 2)
    X_pp = X[mask_pp]
    y_pp = (y[mask_pp] == 2).astype(int)
    mom  = X_pp[:, 4]   # feature 4 = momentum_GeV

    bins_edges  = [0.5, 1.0, 2.0, 5.0, 12.0]
    bins_labels = ["0.5–1.0", "1.0–2.0", "2.0–5.0", ">5.0"]
    bins_center = [0.75, 1.5, 3.5, 7.0]

    auc_bins = []
    n_bins   = []
    MIN_PER_CLASS = 200

    print("  Classificadores por bin de momento (features = dE/dx × 4):")
    for lo, hi, label in zip(bins_edges[:-1], bins_edges[1:], bins_labels):
        m = (mom >= lo) & (mom < hi)
        X_bin = X_pp[m][:, :4]   # só os 4 dE/dx, sem o momento
        y_bin = y_pp[m]

        n_pi = (y_bin == 0).sum()
        n_p  = (y_bin == 1).sum()
        n_total = len(y_bin)

        if n_pi < MIN_PER_CLASS or n_p < MIN_PER_CLASS:
            print(f"    p ∈ [{label}] GeV/c: "
                  f"estatística insuficiente (π={n_pi}, p={n_p}) — ignorado")
            auc_bins.append(np.nan)
            n_bins.append(n_total)
            continue

        # split 70/30 estratificado dentro do bin
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_bin, y_bin, test_size=0.30,
            random_state=RANDOM_STATE, stratify=y_bin
        )
        sw = compute_sample_weight("balanced", y_tr)
        clf_bin = HistGradientBoostingClassifier(**CLF_PARAMS)
        clf_bin.fit(X_tr, y_tr, sample_weight=sw)

        y_prob_bin = clf_bin.predict_proba(X_te)[:, 1]
        auc_val = roc_auc_score(y_te, y_prob_bin)
        auc_bins.append(auc_val)
        n_bins.append(n_total)
        print(f"    p ∈ [{label}] GeV/c: AUC = {auc_val:.4f}  "
              f"(n={n_total:,}: π={n_pi:,}, p={n_p:,})")

    # --- gráfico ---
    valid = [(c, a, n) for c, a, n in zip(bins_center, auc_bins, n_bins)
             if not np.isnan(a)]
    if not valid:
        print("  [AVISO] Nenhum bin com estatística suficiente.")
        return

    cx, av, nv = zip(*valid)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(cx, av, s=[n / 50 for n in nv],
                    c=av, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                    zorder=5, edgecolors="black", linewidths=0.7)
    ax.plot(cx, av, "k--", lw=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="AUC")

    for c, a, n in zip(cx, av, nv):
        ax.annotate(f"AUC={a:.3f}\n(n={n:,})",
                    (c, a), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    ax.axhline(0.5, color="gray", lw=1.2, linestyle=":",
               label="Baseline aleatório (AUC=0.5)")
    ax.axhline(0.9, color="green", lw=1.0, linestyle="--",
               alpha=0.6, label="Limiar de boa separação (AUC=0.9)")
    ax.set_xlabel("Momento (GeV/c)", fontsize=12)
    ax.set_ylabel("AUC (π vs p, só dE/dx)", fontsize=12)
    ax.set_title("AUC do classificador dE/dx por bin de momento\n"
                 "π vs p — features: deposição de energia nos 4 detetores",
                 fontsize=12)
    ax.set_xticks([0.75, 1.5, 3.5, 7.0])
    ax.set_xticklabels(["0.5–1.0", "1.0–2.0", "2.0–5.0", ">5.0"])
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/ml_auc_vs_momentum.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# FASE 2 — Classificador de produção (π vs p, p ∈ [0.5, 2.0] GeV/c)
# ---------------------------------------------------------------------------

P_MIN = 0.5   # GeV/c  — limite inferior do domínio de validade
P_MAX = 2.0   # GeV/c  — limite superior do domínio de validade


def dominio_valido(momentum_gev):
    """
    Verifica se o momento está dentro do domínio de validade do classificador.

    Fora deste intervalo o classificador NÃO deve ser utilizado:
      p < 0.5 GeV/c  → partículas não relativistas; distribuições de Landau
                        mal definidas para todas as espécies.
      p > 2.0 GeV/c  → piões e protões convergem para o mínimo de ionização;
                        dE/dx deixa de ser discriminativo (AUC → 0.5,
                        confirmado pelo gráfico ml_auc_vs_momentum.png).

    Returns
    -------
    numpy bool array — True = válido, False = fora do domínio.
    """
    m = np.asarray(momentum_gev)
    return (m >= P_MIN) & (m <= P_MAX)


def carregar_dados_binario():
    """
    Carrega π e p com p ∈ [P_MIN, P_MAX] GeV/c e EdepDet0_keV > 10.

    Motivação da restrição superior P_MAX = 2.0 GeV/c:
      O gráfico ml_auc_vs_momentum.png mostra AUC = 0.93 no bin [1, 2] GeV/c
      e AUC = 0.72 no bin [2, 5] GeV/c. Acima de 2 GeV/c as curvas de
      Bethe-Bloch convergem rapidamente e nenhum algoritmo consegue separar
      as espécies por dE/dx.

    Limitação conhecida — protões não completamente relativistas:
      A p = 0.5 GeV/c, βγ(p) ≈ 0.53 (longe do MIP).
      A p = 1.0 GeV/c, βγ(p) ≈ 1.07 (ainda em transição).
      A p = 2.0 GeV/c, βγ(p) ≈ 2.13 (razoavelmente relativista).
      A boa discriminação no bin [0.5, 1.0] GeV/c existe parcialmente porque
      protões não-relativistas têm distribuições de Landau mais largas e
      deslocadas, o que ajuda o classificador mas não pela física de Landau
      "pura". Um filtro rigoroso para protões seria p > 1.0 GeV/c (βγ > 1.07),
      mas reduziria substancialmente a amostra disponível.
    """
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
        ((np.abs(pdg) == 211) | (pdg == 2212)) &
        (edep0 > 10) &
        (mom >= P_MIN) &
        (mom <= P_MAX)
    )

    y = (data["particlePDG"][mask] == 2212).astype(int)  # 0=pião, 1=protão
    X = np.column_stack([
        data["EdepDet0_keV"][mask], data["EdepDet1_keV"][mask],
        data["EdepDet2_keV"][mask], data["EdepDet3_keV"][mask],
        data["momentum_GeV"][mask],
    ])

    print(f"  Domínio: p ∈ [{P_MIN}, {P_MAX}] GeV/c  (π e p apenas)")
    print(f"  Total após filtros : {mask.sum():,}")
    print(f"  Piões   (y=0): {(y==0).sum():>8,}  ({100*(y==0).mean():.1f}%)")
    print(f"  Protões (y=1): {(y==1).sum():>8,}  ({100*(y==1).mean():.1f}%)")
    return X, y


def validacao_cruzada_binaria(X, y, n_splits=5):
    """5-fold CV binário."""
    print(f"\n  Validação cruzada binária ({n_splits} folds):")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)
    acc_f, auc_f = [], []
    for fold, (tr, val) in enumerate(cv.split(X, y), 1):
        sw = compute_sample_weight("balanced", y[tr])
        c  = HistGradientBoostingClassifier(**CLF_PARAMS)
        c.fit(X[tr], y[tr], sample_weight=sw)
        acc_f.append(accuracy_score(y[val], c.predict(X[val])))
        auc_f.append(roc_auc_score(y[val], c.predict_proba(X[val])[:, 1]))
        print(f"    Fold {fold}: acc = {acc_f[-1]:.4f}  |  AUC = {auc_f[-1]:.4f}")
    print(f"  Média: acc = {np.mean(acc_f):.4f} ± {np.std(acc_f):.4f}"
          f"   AUC = {np.mean(auc_f):.4f} ± {np.std(auc_f):.4f}")
    return acc_f, auc_f


def plot_roc_binario(clf, X_test, y_test, acc_f, auc_f):
    """ROC + matriz de confusão 2×2."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(fpr, tpr, color="steelblue", lw=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos", fontsize=12)
    ax.set_ylabel("Taxa de verdadeiros positivos", fontsize=12)
    ax.set_title(f"Curva ROC — Classificador de produção\n"
                 f"π vs p,  p ∈ [{P_MIN}, {P_MAX}] GeV/c", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    cv_txt = (f"Validação cruzada (5-fold)\n"
              f"AUC = {np.mean(auc_f):.4f} ± {np.std(auc_f):.4f}\n"
              f"Acc = {np.mean(acc_f):.4f} ± {np.std(acc_f):.4f}")
    ax.text(0.38, 0.08, cv_txt, transform=ax.transAxes, fontsize=9,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))

    ax2 = axes[1]
    cm   = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=["Pião", "Protão"])
    disp.plot(ax=ax2, colorbar=True, cmap="Blues", values_format=".1%")
    ax2.set_title(f"Matriz de confusão (normalizada)\n"
                  f"p ∈ [{P_MIN}, {P_MAX}] GeV/c", fontsize=12)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/ml_binary_roc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    return roc_auc


def plot_cv_binario(acc_f, auc_f):
    """CV scores binários."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    folds = np.arange(1, len(acc_f) + 1)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(folds - w/2, acc_f, w,
                label="Accuracy", color="steelblue", alpha=0.85)
    b2 = ax.bar(folds + w/2, auc_f, w,
                label="AUC", color="darkorange", alpha=0.85)
    ax.bar_label(b1, fmt="%.4f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.4f", fontsize=8, padding=2)
    ax.axhline(np.mean(acc_f), color="steelblue", linestyle="--", lw=1.5,
               label=f"Média acc = {np.mean(acc_f):.4f}")
    ax.axhline(np.mean(auc_f), color="darkorange", linestyle="--", lw=1.5,
               label=f"Média AUC = {np.mean(auc_f):.4f}")
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Validação cruzada 5-fold — classificador de produção\n"
                 f"π vs p,  p ∈ [{P_MIN}, {P_MAX}] GeV/c", fontsize=12)
    ax.set_xticks(folds)
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/ml_binary_cv.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_effpur_binario(clf, X_test, y_test):
    """Curva eficiência-pureza para o classificador binário."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_prob = clf.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(rec, prec, color="steelblue", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Eficiência (recall de protões)", fontsize=12)
    ax.set_ylabel("Pureza (precision de protões)", fontsize=12)
    ax.set_title(f"Curva eficiência–pureza\nπ vs p,  p ∈ [{P_MIN}, {P_MAX}] GeV/c",
                 fontsize=12)
    ax.grid(True, alpha=0.3)

    op_results = []
    for target, col in [(0.90, "green"), (0.95, "orange"), (0.99, "red")]:
        valid = np.where(prec >= target)[0]
        if not len(valid):
            continue
        idx = valid[np.argmax(rec[valid])]
        ax.scatter(rec[idx], prec[idx], s=100, color=col, zorder=5,
                   label=f"Pureza {int(target*100)}%: eff={rec[idx]:.2f}")
        thr = thresholds[idx] if idx < len(thresholds) else 1.0
        op_results.append((target, rec[idx], thr))

    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.3, 1.02)

    ax2 = axes[1]
    ax2.plot(thresholds, prec[:-1], color="darkorange", lw=2, label="Pureza")
    ax2.plot(thresholds, rec[:-1],  color="steelblue",  lw=2, label="Eficiência")
    f1 = 2*prec[:-1]*rec[:-1] / (prec[:-1]+rec[:-1]+1e-12)
    ax2.plot(thresholds, f1, color="green", lw=2, linestyle="--", label="F1-score")
    ax2.axvline(0.5, color="gray", lw=1, linestyle=":", label="Threshold = 0.5")
    ax2.set_xlabel("Threshold de decisão P(protão)", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Pureza, eficiência e F1 vs threshold", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/ml_binary_effpur.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")

    print("  Pontos de operação (π vs p):")
    for pur, eff, thr in op_results:
        print(f"    Pureza {int(pur*100)}%: "
              f"eficiência = {eff:.3f}, threshold = {thr:.3f}")
    return op_results


def plot_features_binario(clf, X_test, y_test):
    """Permutation importance para o classificador binário."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    idx_s = rng.choice(len(y_test),
                       size=min(3000, len(y_test)), replace=False)
    result = permutation_importance(
        clf, X_test[idx_s], y_test[idx_s],
        n_repeats=15, random_state=RANDOM_STATE, scoring="roc_auc",
    )
    imp  = result.importances_mean
    std  = result.importances_std
    idx  = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([FEATURE_NAMES_PT[i] for i in idx], imp[idx],
            xerr=std[idx], color="steelblue", edgecolor="white",
            capsize=4, error_kw={"ecolor": "gray", "lw": 1.5})
    ax.set_xlabel("Queda em AUC (permutation importance)", fontsize=11)
    ax.set_title(f"Importância das variáveis — classificador de produção\n"
                 f"π vs p,  p ∈ [{P_MIN}, {P_MAX}] GeV/c", fontsize=12)
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/ml_binary_features.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_boundary_binario(clf, X, y):
    """Fronteira de decisão 2D para o classificador binário."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    KEV_TO_MEV = 1e-3
    rng = np.random.default_rng(RANDOM_STATE)

    n_sample = 5000
    indices = []
    for cls in [0, 1]:
        idx_cls = np.where(y == cls)[0]
        n_cls = min(int(n_sample * len(idx_cls) / len(y)), len(idx_cls))
        indices.append(rng.choice(idx_cls, size=n_cls, replace=False))
    sel = np.concatenate(indices)
    X_s, y_s = X[sel], y[sel]
    edep_mev, mom_gev = X_s[:, 0] * KEV_TO_MEV, X_s[:, 4]

    n_grid = 250
    xx, yy = np.meshgrid(
        np.linspace(max(edep_mev.min()*0.9, 0.001), edep_mev.max()*1.1, n_grid),
        np.linspace(mom_gev.min()*0.95, mom_gev.max()*1.05, n_grid),
    )
    grid_feat = np.column_stack([
        xx.ravel()/KEV_TO_MEV,
        np.full(xx.size, np.median(X[:, 1])),
        np.full(xx.size, np.median(X[:, 2])),
        np.full(xx.size, np.median(X[:, 3])),
        yy.ravel(),
    ])
    Z = clf.predict_proba(grid_feat)[:, 1].reshape(xx.shape)

    cmap_bg = mcolors.LinearSegmentedColormap.from_list(
        "binary_pid", [(0.85, 0.90, 1.0), (1.0, 0.85, 1.0)]
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(xx, yy, Z, levels=50, cmap=cmap_bg,
                     alpha=0.75, vmin=0, vmax=1)
    plt.colorbar(cf, ax=ax, label="P(protão)")
    ax.contour(xx, yy, Z, levels=[0.5], colors="black",
               linewidths=1.5, linestyles="--")
    ax.scatter(edep_mev[y_s==0], mom_gev[y_s==0],
               c="red", s=5, alpha=0.4,
               label=f"Pião (n={(y_s==0).sum()})")
    ax.scatter(edep_mev[y_s==1], mom_gev[y_s==1],
               c="magenta", s=5, alpha=0.4,
               label=f"Protão (n={(y_s==1).sum()})")
    ax.set_xlabel("dE/dx — Detetor 0 (MeV)", fontsize=12)
    ax.set_ylabel("Momento (GeV/c)", fontsize=12)
    ax.set_title(f"Fronteira de decisão — classificador de produção\n"
                 f"π vs p,  p ∈ [{P_MIN}, {P_MAX}] GeV/c  (5 000 eventos)",
                 fontsize=12)
    ax.legend(fontsize=10, markerscale=3)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/ml_binary_boundary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# 7. Curvas de aprendizagem
# ---------------------------------------------------------------------------

def plot_learning_curves(X, y, suffix="", titulo=""):
    """
    Curvas de aprendizagem: balanced accuracy no treino e na validação cruzada
    em função do tamanho do conjunto de treino.

    Permite detectar overfitting (gap treino/val grande) ou underfitting
    (ambas as curvas baixas). Num classificador bem ajustado as duas curvas
    convergem para um valor elevado com o aumento dos dados.
    """
    from sklearn.model_selection import learning_curve
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_sizes, train_scores, val_scores = learning_curve(
        HistGradientBoostingClassifier(**CLF_PARAMS),
        X, y,
        train_sizes=np.linspace(0.10, 1.0, 8),
        cv=StratifiedKFold(n_splits=5, shuffle=True,
                           random_state=RANDOM_STATE),
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std  = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, tr_mean, "o-", color="steelblue",
            lw=2, label="Treino")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color="steelblue")
    ax.plot(train_sizes, va_mean, "s-", color="darkorange",
            lw=2, label="Validação cruzada (5-fold)")
    ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std,
                    alpha=0.15, color="darkorange")

    ax.set_xlabel("Tamanho do conjunto de treino (eventos)", fontsize=12)
    ax.set_ylabel("Balanced accuracy", fontsize=12)
    t = titulo if titulo else f"Curvas de aprendizagem{' — ' + suffix if suffix else ''}"
    ax.set_title(t, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.50, 1.05)
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/ml_learning_curves{('_' + suffix) if suffix else ''}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# 8. Programa principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Classificador ML para PID  (v3)")
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
    print("  Baseline aleatório (3 classes): 33.3%")
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
    plot_efficiency_purity(clf, X_test, y_test)

    print("\nA gerar curvas de aprendizagem (Fase 1)...")
    plot_learning_curves(
        X, y,
        suffix="fase1",
        titulo="Curvas de aprendizagem — Fase 1 (π, K, p)",
    )

    print("\nA treinar classificadores por bin de momento...")
    plot_auc_vs_momentum(X, y)

    print(f"\nAUC por classe (3 classes, exploratório):")
    for nome, val in zip(CLASS_NAMES, auc_por_classe):
        print(f"  {nome:8s}: {val:.4f}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FASE 2 — Classificador de produção (π vs p, p ∈ [0.5, 2.0] GeV/c)")
    print("=" * 60)

    print("\n[1/5] A carregar dados binários...")
    X_b, y_b = carregar_dados_binario()

    print("\n[2/5] A dividir conjuntos treino/teste (70/30)...")
    X_btr, X_bte, y_btr, y_bte = train_test_split(
        X_b, y_b, test_size=0.30, random_state=RANDOM_STATE, stratify=y_b
    )
    print(f"  Treino: {len(y_btr):,}  |  Teste: {len(y_bte):,}")

    print("\n[3/5] A efectuar validação cruzada binária (5 folds)...")
    acc_fb, auc_fb = validacao_cruzada_binaria(X_b, y_b)

    print("\n[4/5] A treinar classificador de produção...")
    sw_b = compute_sample_weight("balanced", y_btr)
    clf_b = HistGradientBoostingClassifier(**CLF_PARAMS)
    clf_b.fit(X_btr, y_btr, sample_weight=sw_b)
    print("  Treino concluído.")

    print("\n[5/5] A avaliar e gerar gráficos...")

    # Verificar domínio de validade antes de classificar (passo obrigatório em
    # produção: eventos fora de [P_MIN, P_MAX] GeV/c são marcados como
    # "não classificável por dE/dx" e não devem ser passados ao classificador).
    validos_te = dominio_valido(X_bte[:, 4])
    n_fora = int((~validos_te).sum())
    if n_fora > 0:
        print(f"  AVISO: {n_fora} eventos fora do domínio [{P_MIN}, {P_MAX}] GeV/c "
              f"— removidos antes da classificação")
        X_bte = X_bte[validos_te]
        y_bte = y_bte[validos_te]
    else:
        print(f"  Domínio OK: todos os {len(X_bte):,} eventos estão em "
              f"p ∈ [{P_MIN}, {P_MAX}] GeV/c")

    y_bpred = clf_b.predict(X_bte)
    y_bprob = clf_b.predict_proba(X_bte)[:, 1]
    acc_b    = accuracy_score(y_bte, y_bpred)
    bal_b    = balanced_accuracy_score(y_bte, y_bpred)
    auc_b    = roc_auc_score(y_bte, y_bprob)

    print("\n" + "=" * 60)
    print("RESULTADOS — Classificador de produção (conjunto de teste)")
    print("=" * 60)
    print(f"  AUC                  : {auc_b:.4f}")
    print(f"  Balanced accuracy    : {bal_b:.4f}  ({bal_b*100:.2f}%)")
    print(f"  Accuracy             : {acc_b:.4f}  ({acc_b*100:.2f}%)")
    print(f"\n  Domínio de validade  : p ∈ [{P_MIN}, {P_MAX}] GeV/c")
    print("  Fora do domínio: classificar como 'não classificável por dE/dx'")
    print("\nRelatório de classificação:")
    print(classification_report(y_bte, y_bpred,
                                target_names=["Pião", "Protão"], digits=4))

    print("A gerar gráficos da Fase 2...")
    plot_roc_binario(clf_b, X_bte, y_bte, acc_fb, auc_fb)
    plot_cv_binario(acc_fb, auc_fb)
    plot_effpur_binario(clf_b, X_bte, y_bte)
    plot_features_binario(clf_b, X_bte, y_bte)
    plot_boundary_binario(clf_b, X_b, y_b)

    print("\nA gerar curvas de aprendizagem (Fase 2)...")
    plot_learning_curves(
        X_b, y_b,
        suffix="fase2",
        titulo=f"Curvas de aprendizagem — Fase 2 (π vs p, p ∈ [{P_MIN}, {P_MAX}] GeV/c)",
    )

    print("\nDone. Todos os gráficos guardados em:", OUTPUT_DIR)
