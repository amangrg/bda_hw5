#!/usr/bin/env python3
import time, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score, RocCurveDisplay,
    average_precision_score, PrecisionRecallDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = 'spambase'

RAND = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RAND)
N_JOBS = -1

def load_dataset(choice):
    """
    Returns (X, y, name, notes) where X,y are pandas DataFrames/Series.
    """
    ds = fetch_ucirepo(id=94)
    X = ds.data.features.copy()
    y = ds.data.targets.squeeze().copy()
    name = "Spambase (UCI id=94)"
    notes = "57 continuous features; classic email spam classification."
    return X, y, name, notes

X, y, ds_name, ds_notes = load_dataset(dataset)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RAND
)

print("\n=== Deliverable 1: Dataset details ===")
print("Dataset URL:", "https://archive.ics.uci.edu/dataset/94/spambase")
print("Name:", ds_name)
print("Notes:", ds_notes)
print(f"#Instances total: {len(X)}")
print(f"#Training examples: {len(X_train)} | #Test examples: {len(X_test)}")

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("\nNumeric attribute ranges (min..max):")
for c in num_cols[:50]:   # print first 50 if huge
    print(f" - {c}: {X[c].min()} .. {X[c].max()}")
if cat_cols:
    print("\nCategorical attributes (unique counts):")
    for c in cat_cols[:50]:
        print(f" - {c}: {X[c].nunique()} categories")

ex = X_train.head(5).copy()
ex["target"] = y_train.iloc[:5].values
print("\nFive concrete training examples (X | y):")
print(ex.to_string(index=False))

pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', 'passthrough', num_cols)
    ],
    remainder='drop'
)

X_train_proc = pre.fit_transform(X_train)
X_test_proc  = pre.transform(X_test)


tree = DecisionTreeClassifier(random_state=RAND)
grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 6, 10, 16, 24],
    "min_samples_split": [2, 10, 40],
    "min_samples_leaf": [1, 2, 5],
    "ccp_alpha": [0.0, 1e-3]
}
print("\n=== Training Single Decision Tree (with 5-fold CV) ===")
t0 = time.perf_counter()
gs = GridSearchCV(tree, grid, cv=CV, n_jobs=N_JOBS, scoring="accuracy", refit=True)
gs.fit(X_train_proc, y_train)
t_tree = time.perf_counter() - t0

best_tree = gs.best_estimator_
yhat_tree = best_tree.predict(X_test_proc)
acc_tree = accuracy_score(y_test, yhat_tree)

# AUC metrics (esp. valuable for imbalanced sets)
proba_tree = best_tree.predict_proba(X_test_proc)[:,1]
roc_tree = roc_auc_score(y_test, proba_tree)
ap_tree  = average_precision_score(y_test, proba_tree)

print(f"Best hyperparameters: {gs.best_params_}")
print(f"Training time (s): {t_tree:.4f}")
print(f"Test accuracy: {acc_tree:.4f} | ROC-AUC: {roc_tree:.4f} | PR-AUC: {ap_tree:.4f}")

def train_rf(n_trees: int):
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_features='sqrt',
        bootstrap=True,
        random_state=RAND,
        n_jobs=N_JOBS
    )
    t0 = time.perf_counter()
    rf.fit(X_train_proc, y_train)
    t = time.perf_counter() - t0
    yhat = rf.predict(X_test_proc)
    acc = accuracy_score(y_test, yhat)
    proba = rf.predict_proba(X_test_proc)[:,1]
    return rf, t, acc, roc_auc_score(y_test, proba), average_precision_score(y_test, proba)

print("\n=== Training Random Forest (25 trees) ===")
rf25, t_rf25, acc_rf25, roc_rf25, ap_rf25 = train_rf(25)
print(f"Training time (s): {t_rf25:.4f}")
print(f"Test accuracy: {acc_rf25:.4f} | ROC-AUC: {roc_rf25:.4f} | PR-AUC: {ap_rf25:.4f}")

print("\n=== Training Random Forest (50 trees) ===")
rf50, t_rf50, acc_rf50, roc_rf50, ap_rf50 = train_rf(50)
print(f"Training time (s): {t_rf50:.4f}")
print(f"Test accuracy: {acc_rf50:.4f} | ROC-AUC: {roc_rf50:.4f} | PR-AUC: {ap_rf50:.4f}")


print("\n=== Deliverable 4: Per-tree test accuracies (RF-50) ===")
per_tree = []
for idx, est in enumerate(rf50.estimators_):
    yhat_e = est.predict(X_test_proc)
    per_tree.append((idx, accuracy_score(y_test, yhat_e)))
per_tree.sort(key=lambda t: t[1], reverse=True)

print("Top 10 trees (index, accuracy):")
for i, a in per_tree[:10]:
    print(f"  #{i:02d}: {a:.4f}")
print("Bottom 10 trees (index, accuracy):")
for i, a in per_tree[-10:]:
    print(f"  #{i:02d}: {a:.4f}")


print("\n=== Deliverable 3: Accuracy & Training time summary ===")
summary = pd.DataFrame(
    [
        ["Single Decision Tree", t_tree, acc_tree, roc_tree, ap_tree, str(gs.best_params_)],
        ["Random Forest (25 trees)", t_rf25, acc_rf25, roc_rf25, ap_rf25, "n_estimators=25, max_features='sqrt', bootstrap=True"],
        ["Random Forest (50 trees)", t_rf50, acc_rf50, roc_rf50, ap_rf50, "n_estimators=50, max_features='sqrt', bootstrap=True"],
    ],
    columns=["Model", "Train Time (s)", "Accuracy", "ROC-AUC", "PR-AUC", "Config / Key Params"],
)
print(summary.to_string(index=False))


grid_trees = [10, 25, 50, 100, 200, 400]
sweep = []
for n in grid_trees:
    rf, t, acc, roc, ap = train_rf(n)
    sweep.append((n, t, acc, roc, ap))
print("\nExtra: Accuracy/ROC/PR vs #Trees sweep")
for n, t, acc, roc, ap in sweep:
    print(f"  n={n:>3} | time={t:7.3f}s | acc={acc:.4f} | roc={roc:.4f} | pr={ap:.4f}")

plt.figure(figsize=(5,3.5))
vals = y.value_counts().sort_index()
plt.bar(['class 0','class 1'], vals.values)
plt.title(f"{ds_name}: Class balance")
plt.ylabel("Count"); plt.tight_layout(); plt.show()

# Accuracy vs Training Time
labels = ["DT", "RF-25", "RF-50"]
accs = [acc_tree, acc_rf25, acc_rf50]
times = [t_tree, t_rf25, t_rf50]

plt.figure(figsize=(6,4))
plt.bar(labels, accs)
plt.ylim(0.5, 1.0)
plt.title("Test Accuracy"); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,4))
plt.plot(labels, times, marker="o")
plt.title("Training Time (s)"); plt.ylabel("seconds"); plt.tight_layout(); plt.show()

# Per-tree accuracy distribution
ids = [i for i,_ in per_tree]
scores = [a for _,a in per_tree]
plt.figure(figsize=(8,4))
plt.bar(ids, scores)
plt.axhline(np.mean(scores), linestyle="--", color="k", label="mean")
plt.legend(); plt.title("RF-50: Per-tree test accuracy"); plt.tight_layout(); plt.show()

# ROC & PR curves (DT vs RF-50)
proba_rf50 = rf50.predict_proba(X_test_proc)[:,1]
RocCurveDisplay.from_predictions(y_test, proba_tree, name="DT")
RocCurveDisplay.from_predictions(y_test, proba_rf50, name="RF-50")
plt.title("ROC curves"); plt.tight_layout(); plt.show()

PrecisionRecallDisplay.from_predictions(y_test, proba_tree, name="DT")
PrecisionRecallDisplay.from_predictions(y_test, proba_rf50, name="RF-50")
plt.title("Precision-Recall curves"); plt.tight_layout(); plt.show()

# Sweeps: accuracy/time vs trees
plt.figure(figsize=(6,4))
plt.plot([n for n,_,_,_,_ in sweep], [acc for _,_,acc,_,_ in sweep], marker="o")
plt.title("Accuracy vs #Trees"); plt.xlabel("#Trees"); plt.ylim(0.8,1.0); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,4))
plt.plot([n for n,_,_,_,_ in sweep], [t for _,t,_,_,_ in sweep], marker="o")
plt.title("Training Time vs #Trees"); plt.xlabel("#Trees"); plt.ylabel("seconds"); plt.tight_layout(); plt.show()