import sys, time
sys.path.insert(0, ".")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ga_feature_selection import GeneticFeatureSelector
from run_experiment import load_datasets

X, y, n_real = load_datasets(0)["Wine+40 noise"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
est = make_pipeline(StandardScaler(), SVC())
t = time.time()
ga = GeneticFeatureSelector(est, random_state=0, verbose=True).fit(Xtr, ytr)
idx = ga.get_support(indices=True)
print("selected", len(idx), "of", X.shape[1], "| noise kept", int((idx >= n_real).sum()), "| seconds", round(time.time()-t, 1))
for name, cols in [("all", slice(None)), ("GA", idx)]:
    m = est.fit(Xtr[:, cols], ytr)
    print(name, round(accuracy_score(yte, m.predict(Xte[:, cols])), 4))
