"""Reproduce the three experiments from the paper.

  python src/run_experiments.py benchmark   # Table I + Fig. 1 (5-fold CV x 3 seeds, all five models)
  python src/run_experiments.py scarcity    # Fig. 2 (accuracy vs number of labeled samples)
  python src/run_experiments.py confusion   # Fig. 3 (MLP confusion matrix, 5-fold run)
  python src/run_experiments.py all         # everything

Options:
  --synthetic   use random data of the same shape (quick check that the code runs)
  --epochs N    training epochs for the neural models (default 40)
  --seeds N     number of random seeds (default 3, as in the paper)

Results (CSV tables and PNG figures) are written to results/.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from data import load_dataset, make_synthetic
from models import MODEL_NAMES, make_model

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def scale(X_train, X_test):
    """Standardize with training-fold statistics only (no leakage into the test fold)."""
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train).astype(np.float32), scaler.transform(X_test).astype(np.float32)


def benchmark(X, y, seeds, epochs):
    """Stratified 5-fold cross-validation repeated over several seeds (15 runs per model)."""
    rows = []
    for seed in range(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte = scale(X[tr], X[te])
            for name in MODEL_NAMES:
                t0 = time.time()
                pred = make_model(name, seed=seed, epochs=epochs).fit(Xtr, y[tr]).predict(Xte)
                rows.append({"model": name, "seed": seed, "fold": fold,
                             "accuracy": accuracy_score(y[te], pred),
                             "macro_f1": f1_score(y[te], pred, average="macro")})
                print(f"seed {seed} fold {fold} {name:<22} acc={rows[-1]['accuracy']:.4f} "
                      f"f1={rows[-1]['macro_f1']:.4f} ({time.time() - t0:.1f}s)")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(RESULTS_DIR, "benchmark_runs.csv"), index=False)

    table = (runs.groupby("model")[["accuracy", "macro_f1"]].agg(["mean", "std"])
             .reindex(MODEL_NAMES).round(4))
    table.to_csv(os.path.join(RESULTS_DIR, "table1_full_data.csv"))
    print("\nTable I - mean over 5 folds x", seeds, "seeds\n", table)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(MODEL_NAMES, table[("macro_f1", "mean")], yerr=table[("macro_f1", "std")],
           capsize=4, color="#4C72B0")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(min(0.8, table[("macro_f1", "mean")].min() - 0.05), 1.005)
    ax.set_title("Macro F1 by model (5-fold CV x %d seeds)" % seeds)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig1_macro_f1.png"), dpi=150)
    plt.close(fig)


def scarcity(X, y, seeds, epochs, fractions=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0)):
    """Fixed held-out test set; train on a shrinking fraction of the remaining data."""
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    names = ["MLP", "Random Forest", "Autoencoder + softmax"]
    rows = []
    for seed in range(seeds):
        for frac in fractions:
            if frac < 1.0:
                X_tr, _, y_tr, _ = train_test_split(X_pool, y_pool, train_size=frac,
                                                    stratify=y_pool, random_state=seed)
            else:
                X_tr, y_tr = X_pool, y_pool
            Xtr, Xte = scale(X_tr, X_test)
            for name in names:
                pred = make_model(name, seed=seed, epochs=epochs).fit(Xtr, y_tr).predict(Xte)
                rows.append({"model": name, "seed": seed, "n_train": len(y_tr),
                             "accuracy": accuracy_score(y_test, pred)})
                print(f"seed {seed} n_train={len(y_tr):>4} {name:<22} acc={rows[-1]['accuracy']:.4f}")
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["model", "n_train"])["accuracy"].agg(["mean", "std"]).round(4)
    runs.to_csv(os.path.join(RESULTS_DIR, "scarcity_runs.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "scarcity_summary.csv"))
    print("\nLabel scarcity - mean test accuracy\n", summary)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        s = summary.loc[name]
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Number of labeled training samples")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. labeled training samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig2_label_scarcity.png"), dpi=150)
    plt.close(fig)


def confusion(X, y, class_names, epochs):
    """Confusion matrix of the MLP pooled over one 5-fold run on all samples."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = np.empty_like(y)
    for tr, te in skf.split(X, y):
        Xtr, Xte = scale(X[tr], X[te])
        pred[te] = make_model("MLP", seed=0, epochs=epochs).fit(Xtr, y[tr]).predict(Xte)
    cm = confusion_matrix(y, pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(RESULTS_DIR, "fig3_confusion_matrix.csv"))
    print("\nMLP confusion matrix (5-fold, all samples)\n", cm,
          f"\naccuracy={accuracy_score(y, pred):.4f}")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("MLP confusion matrix (5-fold)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig3_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["benchmark", "scarcity", "confusion", "all"])
    p.add_argument("--synthetic", action="store_true", help="use random data of the same shape")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y, class_names = make_synthetic() if args.synthetic else load_dataset()
    print(f"Data: {X.shape[0]} samples x {X.shape[1]} genes, classes {dict(zip(class_names, np.bincount(y)))}")

    if args.experiment in ("benchmark", "all"):
        benchmark(X, y, args.seeds, args.epochs)
    if args.experiment in ("scarcity", "all"):
        scarcity(X, y, args.seeds, args.epochs)
    if args.experiment in ("confusion", "all"):
        confusion(X, y, class_names, args.epochs)


if __name__ == "__main__":
    main()
