import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator,
        population_size=30,
        generations=25,
        crossover_rate=0.8,
        mutation_rate=None,
        elite_size=2,
        tournament_size=3,
        alpha=0.1,
        cv=5,
        scoring="accuracy",
        min_features=1,
        patience=8,
        n_jobs=None,
        random_state=None,
        verbose=False,
    ):
        self.estimator = estimator
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.cv = cv
        self.scoring = scoring
        self.min_features = min_features
        self.patience = patience
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _score(self, mask, X, y):
        key = mask.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        cv_score = cross_val_score(
            clone(self.estimator),
            X[:, mask],
            y,
            cv=self._cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        ).mean()
        result = (cv_score - self.alpha * mask.mean(), cv_score)
        self._cache[key] = result
        return result

    def _repair(self, mask, rng):
        deficit = self.min_features - int(mask.sum())
        if deficit > 0:
            off = np.flatnonzero(~mask)
            mask[rng.choice(off, size=deficit, replace=False)] = True
        return mask

    def _tournament(self, fitness, n_parents, rng):
        draws = rng.integers(0, len(fitness), size=(n_parents, self.tournament_size))
        return draws[np.arange(n_parents), np.argmax(fitness[draws], axis=1)]

    def _offspring(self, parents, n_children, rng):
        n_features = parents.shape[1]
        children = np.empty((n_children, n_features), dtype=bool)
        for i in range(0, n_children, 2):
            a, b = parents[i], parents[(i + 1) % len(parents)]
            if rng.random() < self.crossover_rate:
                swap = rng.random(n_features) < 0.5
                c1, c2 = np.where(swap, a, b), np.where(swap, b, a)
            else:
                c1, c2 = a.copy(), b.copy()
            children[i] = c1
            if i + 1 < n_children:
                children[i + 1] = c2
        return children

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self._cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        self._cache = {}
        pm = self.mutation_rate if self.mutation_rate is not None else min(0.5, 2.0 / n_features)

        density = rng.uniform(0.05, 0.4, size=(self.population_size, 1))
        population = rng.random((self.population_size, n_features)) < density
        population = np.array([self._repair(m, rng) for m in population])

        self.history_ = []
        best_fitness, best_mask, best_cv, stale = -np.inf, None, 0.0, 0

        for generation in range(self.generations):
            scored = np.array([self._score(m, X, y) for m in population])
            fitness = scored[:, 0]
            order = np.argsort(-fitness)
            top = order[0]

            if fitness[top] > best_fitness + 1e-12:
                best_fitness, best_cv = fitness[top], scored[top, 1]
                best_mask = population[top].copy()
                stale = 0
            else:
                stale += 1

            self.history_.append(
                {
                    "generation": generation,
                    "best_fitness": float(best_fitness),
                    "best_cv_score": float(best_cv),
                    "mean_fitness": float(fitness.mean()),
                    "best_n_features": int(best_mask.sum()),
                    "evaluations": len(self._cache),
                }
            )
            if self.verbose:
                h = self.history_[-1]
                print(
                    f"gen {generation:02d} fitness={h['best_fitness']:.4f} "
                    f"cv={h['best_cv_score']:.4f} k={h['best_n_features']} evals={h['evaluations']}"
                )
            if stale >= self.patience:
                break

            n_children = self.population_size - self.elite_size
            parents = population[self._tournament(fitness, n_children, rng)]
            children = self._offspring(parents, n_children, rng)
            children ^= rng.random(children.shape) < pm
            children = np.array([self._repair(m, rng) for m in children])
            population = np.vstack([population[order[: self.elite_size]], children])

        self.support_ = best_mask
        self.best_fitness_ = float(best_fitness)
        self.best_cv_score_ = float(best_cv)
        self.n_evaluations_ = len(self._cache)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        return np.flatnonzero(self.support_) if indices else self.support_
