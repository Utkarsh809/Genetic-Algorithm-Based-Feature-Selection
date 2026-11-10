import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_digits, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ga_feature_selection import GeneticFeatureSelector

METHODS = ["All features", "MI top-k", "RFE", "RF importance", "Genetic Algorithm"]


def classifiers(seed):
    return {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }


def with_noise(X, n_noise, rng):
    return np.hstack([X, rng.normal(size=(X.shape[0], n_noise))])


def load_datasets(seed):
    digits, wine = load_digits(), load_wine()
    keep = train_test_split(
        np.arange(len(digits.target)), train_size=600, stratify=digits.target, random_state=seed
    )[0]
    Xs, ys = make_classification(
        n_samples=500,
        n_features=60,
        n_informative=8,
        n_redundant=6,
        n_clusters_per_class=2,
        class_sep=1.0,
        shuffle=False,
        random_state=seed,
    )
    return {
        "Digits+40 noise": (
            with_noise(digits.data[keep], 40, np.random.default_rng(seed)),
            digits.target[keep],
            digits.data.shape[1],
        ),
        "Wine+40 noise": (
            with_noise(wine.data, 40, np.random.default_rng(seed + 1000)),
            wine.target,
            wine.data.shape[1],
        ),
        "Synthetic-60": (Xs, ys, 14),
    }


def baseline_masks(Xtr, ytr, k, seed):
    n = Xtr.shape[1]
    masks = {"All features": np.ones(n, dtype=bool)}
    masks["MI top-k"] = (
        SelectKBest(partial(mutual_info_classif, random_state=seed), k=k).fit(Xtr, ytr).get_support()
    )
    scaled = StandardScaler().fit_transform(Xtr)
    masks["RFE"] = (
        RFE(LogisticRegression(max_iter=2000), n_features_to_select=k, step=0.1)
        .fit(scaled, ytr)
        .get_support()
    )
    rf = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)
    masks["RF importance"] = (
        SelectFromModel(rf, max_features=k, threshold=-np.inf).fit(Xtr, ytr).get_support()
    )
    return masks


def evaluate(est, Xtr, ytr, Xte, yte, mask):
    model = clone(est).fit(Xtr[:, mask], ytr)
    pred = model.predict(Xte[:, mask])
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def run(args):
    rows, histories = [], {}
    for seed in range(args.seeds):
        for ds_name, (X, y, n_true) in load_datasets(seed).items():
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            for clf_name, est in classifiers(seed).items():
                t0 = time.time()
                ga = GeneticFeatureSelector(
                    est,
                    population_size=args.population,
                    generations=args.generations,
                    alpha=args.alpha,
                    cv=args.cv,
                    patience=args.patience,
                    random_state=seed,
                ).fit(Xtr, ytr)
                ga_time = time.time() - t0
                k = int(ga.support_.sum())
                masks = baseline_masks(Xtr, ytr, k, seed)
                masks["Genetic Algorithm"] = ga.support_
                if seed == 0:
                    histories[(ds_name, clf_name)] = pd.DataFrame(ga.history_)
                for method in METHODS:
                    mask = masks[method]
                    acc, f1 = evaluate(est, Xtr, ytr, Xte, yte, mask)
                    idx = np.flatnonzero(mask)
                    rows.append(
                        {
                            "dataset": ds_name,
                            "classifier": clf_name,
                            "seed": seed,
                            "method": method,
                            "n_features": int(mask.sum()),
                            "noise_selected": int((idx >= n_true).sum()),
                            "accuracy": acc,
                            "f1_macro": f1,
                            "ga_seconds": ga_time if method == "Genetic Algorithm" else np.nan,
                            "ga_evaluations": ga.n_evaluations_ if method == "Genetic Algorithm" else np.nan,
                        }
                    )
                print(
                    f"seed={seed} {ds_name:<22} {clf_name:<12} GA k={k:<3} "
                    f"cv={ga.best_cv_score_:.3f} {ga_time:5.1f}s",
                    flush=True,
                )
    return pd.DataFrame(rows), histories


def summarize(df):
    g = (
        df.groupby(["dataset", "classifier", "method"])
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro=("f1_macro", "mean"),
            n_features=("n_features", "mean"),
            noise_selected=("noise_selected", "mean"),
        )
        .reset_index()
    )
    g["method"] = pd.Categorical(g["method"], METHODS, ordered=True)
    return g.sort_values(["dataset", "classifier", "method"]).reset_index(drop=True)


def plot_accuracy(summary, path):
    datasets = summary["dataset"].unique()
    clfs = summary["classifier"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.4), sharey=False)
    width = 0.8 / len(METHODS)
    colors = ["#9aa5b1", "#6c8ebf", "#82b366", "#d6b656", "#d6336c"]
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        sub = summary[summary["dataset"] == ds]
        for i, m in enumerate(METHODS):
            vals = [sub[(sub["classifier"] == c) & (sub["method"] == m)]["accuracy"].iloc[0] for c in clfs]
            errs = [
                np.nan_to_num(sub[(sub["classifier"] == c) & (sub["method"] == m)]["accuracy_std"].iloc[0])
                for c in clfs
            ]
            ax.bar(np.arange(len(clfs)) + i * width, vals, width, yerr=errs, label=m, color=colors[i], capsize=2)
        ax.set_xticks(np.arange(len(clfs)) + width * (len(METHODS) - 1) / 2)
        ax.set_xticklabels(clfs)
        lo = max(0.0, sub["accuracy"].min() - 0.08)
        ax.set_ylim(lo, 1.0)
        ax.set_title(ds)
        ax.set_ylabel("Test accuracy")
        ax.grid(axis="y", alpha=0.25)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_convergence(histories, path):
    datasets = sorted({k[0] for k in histories})
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.2 * len(datasets), 6.4), squeeze=False)
    for j, ds in enumerate(datasets):
        for (d, clf), h in histories.items():
            if d != ds:
                continue
            axes[0, j].plot(h["generation"], h["best_cv_score"], marker="o", ms=3, label=clf)
            axes[1, j].plot(h["generation"], h["best_n_features"], marker="o", ms=3, label=clf)
        axes[0, j].set_title(ds)
        axes[0, j].set_ylabel("Best CV accuracy")
        axes[1, j].set_ylabel("Selected features")
        axes[1, j].set_xlabel("Generation")
        for r in range(2):
            axes[r, j].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reduction(summary, path):
    ga = summary[summary["method"] == "Genetic Algorithm"].copy()
    base = summary[summary["method"] == "All features"].copy()
    key = ["dataset", "classifier"]
    m = base.merge(ga, on=key, suffixes=("_all", "_ga"))
    labels = m["dataset"].str.slice(0, 14) + "\n" + m["classifier"]
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(m)), 4.2))
    ax.bar(x - 0.2, m["n_features_all"], 0.4, label="All features", color="#9aa5b1")
    ax.bar(x + 0.2, m["n_features_ga"], 0.4, label="GA-selected", color="#d6336c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Feature count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--population", type=int, default=30)
    p.add_argument("--generations", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.population, args.generations, args.patience, args.cv = 1, 16, 8, 4, 3

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, histories = run(args)
    summary = summarize(df)

    df.to_csv(out / "results_raw.csv", index=False)
    summary.to_csv(out / "results_summary.csv", index=False)
    pd.concat(
        [h.assign(dataset=d, classifier=c) for (d, c), h in histories.items()]
    ).to_csv(out / "ga_history.csv", index=False)

    plot_accuracy(summary, out / "accuracy_comparison.png")
    plot_convergence(histories, out / "ga_convergence.png")
    plot_reduction(summary, out / "feature_reduction.png")

    view = summary.copy()
    view["accuracy"] = view["accuracy"].map("{:.3f}".format)
    view["f1_macro"] = view["f1_macro"].map("{:.3f}".format)
    view["n_features"] = view["n_features"].map("{:.1f}".format)
    view["noise_selected"] = view["noise_selected"].map("{:.1f}".format)
    view = view.drop(columns="accuracy_std")
    (out / "results_summary.md").write_text(view.to_markdown(index=False))
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
