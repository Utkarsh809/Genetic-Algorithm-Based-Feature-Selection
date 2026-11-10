"""Pretty-print experiment results in the terminal."""

import os, sys

# Fix encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from tabulate import tabulate

OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")

# ── ANSI helpers ──
BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
DIM   = "\033[2m"
RESET = "\033[0m"

def banner(text):
    w = 70
    print(f"\n{CYAN}{'═' * w}")
    print(f"  {BOLD}{text}{RESET}{CYAN}")
    print(f"{'═' * w}{RESET}\n")

def section(text):
    print(f"\n{YELLOW}{BOLD}▸ {text}{RESET}")
    print(f"{YELLOW}{'─' * 60}{RESET}")

# ── Load data ──
raw_path = os.path.join(OUTPUTS, "results_summary.csv")
if not os.path.exists(raw_path):
    print("No results found. Run  python run_experiment.py  first.")
    sys.exit(1)

df = pd.read_csv(raw_path)

# ── Enable ANSI on Windows ──
if sys.platform == "win32":
    os.system("")  # enables VT100 escape sequences

banner("GENETIC ALGORITHM FEATURE SELECTION — RESULTS")

# ── Per-dataset summary ──
for dataset in df["dataset"].unique():
    section(dataset)
    sub = df[df["dataset"] == dataset][["classifier", "method", "accuracy", "f1_macro", "n_features", "noise_selected"]].copy()

    # highlight best accuracy per classifier
    rows = []
    for clf in sub["classifier"].unique():
        clf_df = sub[sub["classifier"] == clf]
        best_acc = clf_df["accuracy"].max()
        for _, r in clf_df.iterrows():
            acc_str = f"{r['accuracy']:.3f}"
            if r["accuracy"] == best_acc:
                acc_str = f"{GREEN}{BOLD}{acc_str} ★{RESET}"
            rows.append([
                r["classifier"],
                r["method"],
                acc_str,
                f"{r['f1_macro']:.3f}",
                int(r["n_features"]),
                int(r["noise_selected"]),
            ])

    print(tabulate(rows,
                   headers=["Classifier", "Method", "Accuracy", "F1", "Features", "Noise Kept"],
                   tablefmt="simple_grid",
                   colalign=("left", "left", "right", "right", "right", "right")))
    print()

# ── Overall winner board ──
section("BEST METHOD PER DATASET × CLASSIFIER")
best = df.loc[df.groupby(["dataset", "classifier"])["accuracy"].idxmax()]
best = best[["dataset", "classifier", "method", "accuracy", "n_features"]].copy()
best["accuracy"] = best["accuracy"].apply(lambda x: f"{GREEN}{BOLD}{x:.3f}{RESET}")
print(tabulate(best.values.tolist(),
               headers=["Dataset", "Classifier", "Best Method", "Accuracy", "Features"],
               tablefmt="simple_grid"))

# ── GA convergence stats ──
hist_path = os.path.join(OUTPUTS, "ga_history.csv")
if os.path.exists(hist_path):
    section("GA CONVERGENCE SUMMARY")
    hist = pd.read_csv(hist_path)
    conv = hist.groupby(["dataset", "classifier"]).agg(
        generations=("generation", "max"),
        final_fitness=("best_fitness", "last"),
        final_cv=("best_cv_score", "last"),
        final_k=("best_n_features", "last"),
    ).reset_index()
    conv["final_fitness"] = conv["final_fitness"].round(4)
    conv["final_cv"] = conv["final_cv"].round(4)
    conv["final_k"] = conv["final_k"].round(1)
    conv["generations"] = conv["generations"].astype(int)
    print(tabulate(conv.values.tolist(),
                   headers=["Dataset", "Classifier", "Generations", "Fitness", "CV Score", "Features"],
                   tablefmt="simple_grid"))

print(f"\n{DIM}Output files: {OUTPUTS}{RESET}\n")
