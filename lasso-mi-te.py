import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import mutual_info_classif

# User parameter: Number of top features to select
num_features = 30

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)
print(f"MNIST dataset shape: {X.shape}")

# 1. LASSO-based feature selection
print(f"\nPerforming LASSO-based feature selection for top {num_features} features...")
# Apply LASSO with a small alpha
lasso = Lasso(alpha=0.01, max_iter=1000, random_state=42)
lasso.fit(X, y)

# Get feature importance from LASSO coefficients
lasso_importance = np.abs(lasso.coef_)
top_lasso_indices = np.argsort(-lasso_importance)[:num_features]
X_lasso_top = X[:, top_lasso_indices]
print(f"LASSO top features dataset shape: {X_lasso_top.shape}")

# Display top 5 features from LASSO (or fewer if num_features < 5)
print(f"\nTop {min(5, num_features)} LASSO most influential original features:")
for i in range(min(5, num_features)):
    feature_idx = top_lasso_indices[i]
    importance = lasso_importance[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")

# 2. Mutual Information-based feature selection
print(f"\nPerforming Mutual Information-based feature selection for top {num_features} features...")
mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
top_mi_indices = np.argsort(-mi_scores)[:num_features]
X_mi_top = X[:, top_mi_indices]
print(f"Mutual Information top features dataset shape: {X_mi_top.shape}")

# Display top 5 features from Mutual Information (or fewer if num_features < 5)
print(f"\nTop {min(5, num_features)} Mutual Information most influential original features:")
for i in range(min(5, num_features)):
    feature_idx = top_mi_indices[i]
    importance = mi_scores[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")

# 3. Transfer Entropy-based feature selection
print(f"\nPerforming Transfer Entropy-based feature selection for top {num_features} features...")

def estimate_transfer_entropy(X, y, bin_count=5, lag=1):
    """
    Estimate transfer entropy from each feature to the target.
    Transfer entropy measures directed information flow.
    """
    n_samples, n_features = X.shape
    
    # Use a smaller random subset for efficiency if needed
    max_samples = 5000
    if n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        X = X[indices]
        y = y[indices]
        n_samples = max_samples
    
    # Bin the data for entropy estimation
    X_bins = np.linspace(np.min(X), np.max(X), bin_count + 1)
    binned_X = np.digitize(X, X_bins)
    
    # For classification targets, we already have discrete values
    unique_y = np.unique(y)
    binned_y = np.zeros_like(y)
    for i, val in enumerate(unique_y):
        binned_y[y == val] = i
    
    te_scores = np.zeros(n_features)
    for i in range(n_features):
        # Calculate how much feature i influences the target
        # We use present features to predict future target
        X_present = binned_X[:-lag, i]
        y_present = binned_y[:-lag]
        y_future = binned_y[lag:]
        
        # Calculate mutual information between feature and future target
        # I(X_present; y_future)
        contingency_xy = np.histogram2d(X_present, y_future, 
                                      bins=[bin_count, len(unique_y)])[0]
        if contingency_xy.sum() > 0:
            # Calculate mutual information
            contingency_xy_norm = contingency_xy / contingency_xy.sum()
            px = contingency_xy.sum(axis=1) / contingency_xy.sum()
            py_future = contingency_xy.sum(axis=0) / contingency_xy.sum()
            outer_prod_xy = np.outer(px, py_future)
            
            # Avoid log(0)
            mask_xy = contingency_xy_norm > 0
            mi_xy = np.sum(contingency_xy_norm[mask_xy] * np.log(contingency_xy_norm[mask_xy] / outer_prod_xy[mask_xy]))
            te_scores[i] += mi_xy
    
    return te_scores

# For efficiency, first reduce dimensionality by selecting features with variance above threshold
var_threshold = np.percentile(np.var(X, axis=0), 50)  # Top 50% by variance
high_var_indices = np.where(np.var(X, axis=0) > var_threshold)[0]
X_var_reduced = X[:, high_var_indices]

# Calculate transfer entropy scores on high variance features to improve efficiency
print(f"Calculating transfer entropy on {X_var_reduced.shape[1]} high variance features...")
te_scores_reduced = estimate_transfer_entropy(X_var_reduced, y)

# Map scores back to original feature indices
te_scores = np.zeros(X.shape[1])
te_scores[high_var_indices] = te_scores_reduced

top_te_indices = np.argsort(-te_scores)[:num_features]
X_te_top = X[:, top_te_indices]
print(f"Transfer Entropy top features dataset shape: {X_te_top.shape}")

# Display top 5 features from Transfer Entropy (or fewer if num_features < 5)
print(f"\nTop {min(5, num_features)} Transfer Entropy most influential original features:")
for i in range(min(5, num_features)):
    feature_idx = top_te_indices[i]
    importance = te_scores[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")

# Cross-validation with 5 folds
print("\nPerforming cross-validation...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Linear classifier (LogisticRegression) for LASSO-selected features
print(f"\nCross-validating LASSO-selected features with Linear Classifier...")
linear_clf = LogisticRegression(max_iter=1000, random_state=42)
scores_lasso_linear = cross_val_score(linear_clf, X_lasso_top, y, cv=cv, scoring='accuracy')
print(f"LASSO top {num_features} features (with Linear Classifier) - CV Accuracy: {scores_lasso_linear.mean():.4f} ± {scores_lasso_linear.std():.4f}")

# Random Forest for all feature selection methods
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate LASSO-selected features with Random Forest
print(f"\nCross-validating LASSO-selected features with Random Forest...")
scores_lasso_rf = cross_val_score(rf, X_lasso_top, y, cv=cv, scoring='accuracy')
print(f"LASSO top {num_features} features (with RF) - CV Accuracy: {scores_lasso_rf.mean():.4f} ± {scores_lasso_rf.std():.4f}")

# Evaluate Mutual Information selected features with Random Forest
print(f"\nCross-validating Mutual Information-selected features with Random Forest...")
scores_mi_rf = cross_val_score(rf, X_mi_top, y, cv=cv, scoring='accuracy')
print(f"MI top {num_features} features (with RF) - CV Accuracy: {scores_mi_rf.mean():.4f} ± {scores_mi_rf.std():.4f}")

# Evaluate Transfer Entropy selected features with Random Forest
print(f"\nCross-validating Transfer Entropy-selected features with Random Forest...")
scores_te_rf = cross_val_score(rf, X_te_top, y, cv=cv, scoring='accuracy')
print(f"TE top {num_features} features (with RF) - CV Accuracy: {scores_te_rf.mean():.4f} ± {scores_te_rf.std():.4f}")

# Create a summary of results
print("\n=== SUMMARY OF RESULTS ===")
results = [
    ("LASSO (with Linear Classifier)", scores_lasso_linear.mean(), scores_lasso_linear.std()),
    ("LASSO (with Random Forest)", scores_lasso_rf.mean(), scores_lasso_rf.std()),
    ("Mutual Information (with Random Forest)", scores_mi_rf.mean(), scores_mi_rf.std()),
    ("Transfer Entropy (with Random Forest)", scores_te_rf.mean(), scores_te_rf.std())
]

# Sort methods by accuracy
results.sort(key=lambda x: x[1], reverse=True)

# Print rankings
print("\nRanking of methods by accuracy:")
for i, (method, mean_acc, std_acc) in enumerate(results, 1):
    print(f"{i}. {method}: {mean_acc:.4f} ± {std_acc:.4f}")

print("\nAnalysis complete!")
