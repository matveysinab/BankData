"""
Credit Card Fraud Detection using Random Forest Classifier.

Dataset: Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud)
- 284,807 transactions, only 492 (0.17%) are fraudulent (heavily imbalanced)

This script compares three imbalance-handling strategies:
  1. class_weight='balanced'
  2. Random undersampling of the majority class
  3. SMOTE oversampling of the minority class
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    RocCurveDisplay,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = 12
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data(path: str = "creditcard.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    fraud_pct = df["Class"].mean() * 100
    print(f"Fraud transactions: {df['Class'].sum()} ({fraud_pct:.4f}%)")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


# ---------------------------------------------------------------------------
# 2. Balancing strategies
# ---------------------------------------------------------------------------

def strategy_class_weight(X_train, y_train):
    """No resampling — pass class_weight='balanced' to the classifier."""
    return X_train, y_train, {"class_weight": "balanced"}


def strategy_undersampling(X_train, y_train):
    """Randomly downsample the majority class to match the minority count."""
    df = pd.concat([X_train, y_train], axis=1)
    majority = df[df["Class"] == 0]
    minority = df[df["Class"] == 1]
    majority_down = resample(majority, replace=False,
                             n_samples=len(minority),
                             random_state=RANDOM_STATE)
    df_balanced = pd.concat([majority_down, minority])
    X_bal = df_balanced.drop(columns=["Class"])
    y_bal = df_balanced["Class"]
    return X_bal, y_bal, {}


def strategy_smote(X_train, y_train):
    """Oversample minority class with SMOTE."""
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res, {}


STRATEGIES = {
    "class_weight": strategy_class_weight,
    "undersampling": strategy_undersampling,
    "SMOTE": strategy_smote,
}


# ---------------------------------------------------------------------------
# 3. Train & evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, y_train, X_test, y_test, rf_kwargs):
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **rf_kwargs,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    return clf, y_pred, y_prob, roc_auc, ap, cm, report


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(ax, cm, title):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Normal", "Fraud"],
           yticklabels=["Normal", "Fraud"],
           ylabel="True label", xlabel="Predicted label",
           title=title)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")


def plot_results(results: dict, X_test, y_test):
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("Credit Card Fraud Detection — Random Forest Comparison",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = ["steelblue", "darkorange", "forestgreen"]
    strategy_names = list(results.keys())

    # Row 0: confusion matrices
    for col, (name, color) in enumerate(zip(strategy_names, colors)):
        ax = fig.add_subplot(gs[0, col])
        cm = results[name]["cm"]
        plot_confusion_matrix(ax, cm, f"Confusion Matrix\n({name})")

    # Row 1: ROC curves (all on one ax) + PR curves (all on one ax) + bar summary
    ax_roc = fig.add_subplot(gs[1, 0])
    ax_pr = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[1, 2])

    bar_names, bar_roc, bar_ap = [], [], []
    for name, color in zip(strategy_names, colors):
        y_prob = results[name]["y_prob"]
        RocCurveDisplay.from_predictions(
            y_test, y_prob,
            name=f"{name} (AUC={results[name]['roc_auc']:.4f})",
            ax=ax_roc, color=color)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = results[name]["ap"]
        ax_pr.plot(rec, prec, color=color,
                   label=f"{name} (AP={ap:.4f})")
        bar_names.append(name)
        bar_roc.append(results[name]["roc_auc"])
        bar_ap.append(ap)

    ax_roc.set_title("ROC Curve")
    ax_roc.legend(fontsize=8)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(fontsize=8)

    x = np.arange(len(bar_names))
    w = 0.35
    ax_bar.bar(x - w / 2, bar_roc, w, label="ROC-AUC", color="steelblue")
    ax_bar.bar(x + w / 2, bar_ap, w, label="Avg Precision", color="darkorange")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_names, rotation=15, ha="right")
    ax_bar.set_ylim(0.8, 1.0)
    ax_bar.set_title("Metric Comparison")
    ax_bar.legend(fontsize=8)
    for i, (r, a) in enumerate(zip(bar_roc, bar_ap)):
        ax_bar.text(i - w / 2, r + 0.001, f"{r:.4f}", ha="center", fontsize=7)
        ax_bar.text(i + w / 2, a + 0.001, f"{a:.4f}", ha="center", fontsize=7)

    # Row 2: calibration curves
    for col, (name, color) in enumerate(zip(strategy_names, colors)):
        ax = fig.add_subplot(gs[2, col])
        CalibrationDisplay.from_predictions(
            y_test, results[name]["y_prob"],
            n_bins=10, name=name, ax=ax, color=color)
        ax.set_title(f"Calibration Curve\n({name})")

    plt.savefig("fraud_detection_results.png", dpi=150, bbox_inches="tight")
    print("Saved: fraud_detection_results.png")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(clf, feature_names, strategy_name):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(indices)),
           importances[indices],
           color="steelblue")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Top-20 Feature Importances ({strategy_name})")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{strategy_name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: feature_importance_{strategy_name}.png")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    # --- Load ---
    X, y = load_data("creditcard.csv")

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Fraud in test: {y_test.sum()} ({y_test.mean()*100:.4f}%)\n")

    results = {}
    best_clf = None
    best_strategy = None

    for name, strategy_fn in STRATEGIES.items():
        print(f"{'='*60}")
        print(f"Strategy: {name}")
        print(f"{'='*60}")

        X_bal, y_bal, rf_kwargs = strategy_fn(X_train, y_train)
        print(f"  Training samples: {len(X_bal)} "
              f"(fraud: {y_bal.sum()}, normal: {(y_bal==0).sum()})")

        clf, y_pred, y_prob, roc_auc, ap, cm, report = train_and_evaluate(
            X_bal, y_bal, X_test, y_test, rf_kwargs
        )

        print(f"  ROC-AUC : {roc_auc:.6f}")
        print(f"  Avg Prec: {ap:.6f}")
        print(f"\n{report}\n")

        results[name] = dict(
            clf=clf, y_pred=y_pred, y_prob=y_prob,
            roc_auc=roc_auc, ap=ap, cm=cm
        )

        if best_clf is None or roc_auc > results[best_strategy]["roc_auc"]:
            best_clf = clf
            best_strategy = name

    # --- Summary table ---
    print("\n=== SUMMARY ===")
    summary = pd.DataFrame(
        {name: {"ROC-AUC": r["roc_auc"], "Avg Precision": r["ap"]}
         for name, r in results.items()}
    ).T
    print(summary.to_string())
    print(f"\nBest strategy by ROC-AUC: {best_strategy}")

    # --- Plots ---
    plot_results(results, X_test, y_test)
    plot_feature_importance(best_clf, list(X.columns), best_strategy)


if __name__ == "__main__":
    main()
