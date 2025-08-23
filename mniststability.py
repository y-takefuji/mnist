# Python code that:
# - Loads MNIST (70,000 samples, 784 features)
# - Performs feature selection using: RandomForest, XGBoost, Feature Agglomeration, HVGS, Spearman (rho with p-values)
# - For each selector:
#     1) Rank on full data, select top 30
#     2) Remove the single highest-ranked feature, rerank on reduced data, select top 29
#     3) Compute stability of top-10 ranking orders between full and reduced rankings
#     4) Cross-validate RandomForest on top-30 features (global selection, as before)
#     5) Cross-validate with Spearman selection IN-FOLD (pipeline) and RF as classifier
# - Additionally prints top-10 features from the top-30 set and top-29 set for Spearman specifically
# Constraints:
# - Exactly 100 trees for both RF and XGBoost
# - No scaling, no normalization, and avoid fit_transform (use fit where applicable)

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone

# ----------------------------
# Data
# ----------------------------

def load_mnist_70k_784():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)  # ensure integer classes 0..9
    return X, y

# ----------------------------
# Utilities
# ----------------------------

def stability_spearman_on_ranks(order1, order2, k=10):
    top1 = order1[:k]
    top2 = order2[:k]
    union = list(dict.fromkeys(list(top1) + list(top2)))
    rank1, rank2 = [], []
    for f in union:
        r1 = np.where(order1 == f)[0][0] + 1 if f in top1 else k + 1
        r2 = np.where(order2 == f)[0][0] + 1 if f in top2 else k + 1
        rank1.append(r1)
        rank2.append(r2)
    rho, _ = spearmanr(rank1, rank2)
    return float(rho)

def rank_indices_from_scores(scores):
    return np.argsort(-scores)

def cross_validate_rf(X, y, random_state=42, n_splits=5):
    clf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(accuracy_score), n_jobs=-1)
    return scores.mean(), scores.std()

# ----------------------------
# Feature selection scoring functions (global ranking)
# ----------------------------

def fs_random_forest(X, y, random_state=42):
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf.feature_importances_

def fs_xgboost(X, y, random_state=42):
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )
    xgb.fit(X, y)
    booster = xgb.get_booster()
    gain_dict = booster.get_score(importance_type="gain")  # keys like 'f0'
    n_features = X.shape[1]
    gains = np.zeros(n_features, dtype=float)
    for k, v in gain_dict.items():
        idx = int(k[1:])
        if 0 <= idx < n_features:
            gains[idx] = v
    return gains

def fs_feature_agglomeration(X, n_clusters=100):
    fa = FeatureAgglomeration(n_clusters=n_clusters, linkage="ward")
    fa.fit(X)
    labels = fa.labels_
    scores = np.zeros(X.shape[1], dtype=float)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        block = X[:, idx]
        centroid = block.mean(axis=1)
        c_std = centroid.std(ddof=1)
        if c_std == 0:
            corr_abs = np.zeros(len(idx))
        else:
            x_mean = block.mean(axis=0)
            x_std = block.std(axis=0, ddof=1)
            cov = ((block - x_mean) * (centroid[:, None] - centroid.mean())).sum(axis=0) / (block.shape[0] - 1)
            denom = x_std * c_std
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.where(denom > 0, cov / denom, 0.0)
            corr_abs = np.abs(corr)
        scores[idx] = corr_abs
    return scores

def fs_hvgs(X):
    return X.var(axis=0, ddof=1)

def fs_spearman_pvalue(X, y):
    classes = np.unique(y)
    y_bin = np.zeros((X.shape[0], len(classes)), dtype=float)
    for i, c in enumerate(classes):
        y_bin[:, i] = (y == c).astype(float)
    scores = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        xj = X[:, j]
        best = 0.0
        for i in range(y_bin.shape[1]):
            rho, p = spearmanr(xj, y_bin[:, i])
            if np.isnan(rho) or np.isnan(p):
                continue
            s = abs(rho) * (-np.log10(max(p, 1e-300)))
            if s > best:
                best = s
        scores[j] = best
    return scores

# ----------------------------
# In-fold Spearman selector (for CV pipeline)
# ----------------------------

class SpearmanTopKSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=30):
        self.k = int(k)
        self.top_indices_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("SpearmanTopKSelector requires y for supervised scoring.")
        classes = np.unique(y)
        y_bin = np.zeros((X.shape[0], len(classes)), dtype=float)
        for i, c in enumerate(classes):
            y_bin[:, i] = (y == c).astype(float)
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j]
            best = 0.0
            for i in range(y_bin.shape[1]):
                rho, p = spearmanr(xj, y_bin[:, i])
                if np.isnan(rho) or np.isnan(p):
                    continue
                s = abs(rho) * (-np.log10(max(p, 1e-300)))
                if s > best:
                    best = s
            scores[j] = best
        self.top_indices_ = np.argsort(-scores)[: self.k]
        return self

    def transform(self, X):
        if self.top_indices_ is None:
            raise RuntimeError("Must call fit before transform.")
        return X[:, self.top_indices_]

# ----------------------------
# Run a single method
# ----------------------------

def run_method(name, scorer_func, X, y, top_k_full=30, top_k_reduced=29, random_state=42):
    results = {}

    # 1) Score on full set
    if name in ("random_forest", "xgboost", "spearman"):
        scores_full = scorer_func(X, y)
    else:
        scores_full = scorer_func(X)
    ranks_full = rank_indices_from_scores(scores_full)
    top_full = ranks_full[:top_k_full]
    highest_feature = ranks_full[0]

    # 2) Cross-validate RF on top-30 (global selection)
    X_top30 = X[:, top_full]
    cv_mean, cv_std = cross_validate_rf(X_top30, y, random_state=random_state)

    # 3) Remove highest feature, rescore, select top 29
    keep_mask = np.ones(X.shape[1], dtype=bool)
    keep_mask[highest_feature] = False
    X_reduced = X[:, keep_mask]

    if name in ("random_forest", "xgboost", "spearman"):
        scores_red = scorer_func(X_reduced, y)
    else:
        scores_red = scorer_func(X_reduced)

    ranks_red_local = rank_indices_from_scores(scores_red)
    top_red_local = ranks_red_local[:top_k_reduced]
    reduced_to_original = np.where(keep_mask)[0]
    top_red = reduced_to_original[top_red_local]

    # 4) Stability of top-10 ranking orders
    order_full = ranks_full
    order_red = reduced_to_original[ranks_red_local]
    stability = stability_spearman_on_ranks(order_full, order_red, k=10)

    # 5) In-fold Spearman selection with RF (only for Spearman method requested)
    spearman_cv_mean = None
    spearman_cv_std = None
    if name == "spearman":
        selector = SpearmanTopKSelector(k=top_k_full)
        clf = RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1
        )
        # Manual CV to use in-fold selection
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        fold_scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            sel = clone(selector)
            sel.fit(X_tr, y_tr)
            X_tr_sel = sel.transform(X_tr)
            X_te_sel = sel.transform(X_te)
            clf_fold = clone(clf)
            clf_fold.fit(X_tr_sel, y_tr)
            acc = accuracy_score(y_te, clf_fold.predict(X_te_sel))
            fold_scores.append(acc)
        fold_scores = np.asarray(fold_scores, dtype=float)
        spearman_cv_mean = float(fold_scores.mean())
        spearman_cv_std = float(fold_scores.std())

    # Pack
    results["method"] = name
    results["top30_indices_full"] = top_full
    results["highest_feature_full"] = highest_feature
    results["top29_indices_reduced"] = top_red
    results["stability_top10_spearman"] = stability
    results["rf_cv_mean_on_top30"] = cv_mean
    results["rf_cv_std_on_top30"] = cv_std
    if spearman_cv_mean is not None:
        results["spearman_infold_rf_cv_mean_on_top30"] = spearman_cv_mean
        results["spearman_infold_rf_cv_std_on_top30"] = spearman_cv_std

    return results

# ----------------------------
# Main
# ----------------------------

def main():
    X, y = load_mnist_70k_784()

    methods = [
        ("random_forest", fs_random_forest),
        ("xgboost", fs_xgboost),
        ("feature_agglomeration", lambda X_, y_=None: fs_feature_agglomeration(X_)),
        ("hvgs", lambda X_, y_=None: fs_hvgs(X_)),
        ("spearman", fs_spearman_pvalue),
    ]

    all_results = []
    for name, func in methods:
        print(f"\nRunning method: {name}")
        res = run_method(name, func, X, y, top_k_full=30, top_k_reduced=29, random_state=42)
        all_results.append(res)

        print(f"Highest-ranked feature (full): {res['highest_feature_full']}")
        print(f"Top-30 features (full) first 10: {res['top30_indices_full'][:10]}")
        print(f"Top-29 features (reduced) first 10: {res['top29_indices_reduced'][:10]}")
    #    print(f"Stability (Spearman on ranks) for top-10: {res['stability_top10_spearman']:.4f}")
        print(f"RandomForest CV accuracy on top-30 (global selection): {res['rf_cv_mean_on_top30']:.4f} ± {res['rf_cv_std_on_top30']:.4f}")

        # Additional prints specifically for Spearman:
        if name == "spearman":
            top10_full = res["top30_indices_full"][:10]
            top10_reduced = res["top29_indices_reduced"][:10]
            print(f"[Spearman] Top-10 of top-30 (full): {top10_full}")
            print(f"[Spearman] Top-10 of top-29 (reduced): {top10_reduced}")
            if "spearman_infold_rf_cv_mean_on_top30" in res:
                print(
                    f"[Spearman] In-fold selection CV (RF on top-30): "
                    f"{res['spearman_infold_rf_cv_mean_on_top30']:.4f} ± {res['spearman_infold_rf_cv_std_on_top30']:.4f}"
                )

    return all_results

if __name__ == "__main__":
    main()
