# Genetic Algorithm-Based Feature Selection

Wrapper-style feature selection with a genetic algorithm, benchmarked against filter, embedded and recursive baselines across three classifiers and three noisy datasets.

## Method

- Chromosome: binary mask over features
- Fitness: `CV score - alpha * (selected / total)` using stratified k-fold on the training split only
- Selection: tournament; crossover: uniform; mutation: per-bit flip (default `2 / n_features`); elitism; early stopping on stagnation
- Fitness cache keyed by mask, so repeated chromosomes cost nothing
- Test data is never seen by the GA

## Usage

```bash
pip install -r requirements.txt
python run_experiment.py            # full: 3 seeds, pop 30, up to 25 generations
python run_experiment.py --quick    # smoke test
```

```python
from ga_feature_selection import GeneticFeatureSelector

ga = GeneticFeatureSelector(estimator, population_size=30, generations=25, alpha=0.1, random_state=0)
X_train_sel = ga.fit_transform(X_train, y_train)
X_test_sel = ga.transform(X_test)
ga.get_support(indices=True)
```

## Experiment

Datasets: Digits (600-sample subset, 64 pixels) + 40 noise features, Wine + 40 noise features, synthetic (60 features, 14 informative/redundant).
Classifiers: KNN, SVM (RBF), Logistic Regression.
Baselines (given the same feature budget `k` as the GA): all features, mutual-information top-k, RFE, random-forest importance top-k.
Reported over 3 seeds on a 30% held-out test split.

## Outputs

`outputs/`: `results_raw.csv`, `results_summary.csv`, `results_summary.md`, `ga_history.csv`, `accuracy_comparison.png`, `ga_convergence.png`, `feature_reduction.png`
