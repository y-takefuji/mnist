import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# User parameter: Number of top features to select
num_features = 30

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Take only digits 0-9 (should be all of them in MNIST)
print("Preprocessing dataset...")
X = np.array(X)
y = np.array(y)
print(f"Original MNIST dataset shape: {X.shape}")

# Normalize pixel values
X = X / 255.0

# For more efficient feature selection, use a subsample
X_subsample, _, y_subsample, _ = train_test_split(X, y, test_size=0.9, random_state=42)
print(f"Using subsample of shape {X_subsample.shape} for feature selection...")

# 1. LASSO-based feature selection
print(f"\nPerforming LASSO-based feature selection for top {num_features} features...")
# Scale features for LASSO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subsample)

# Apply LASSO with a small alpha
lasso = Lasso(alpha=0.01, max_iter=1000, random_state=42)
lasso.fit(X_scaled, y_subsample)

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
mi_scores = mutual_info_classif(X_subsample, y_subsample, discrete_features=False, random_state=42)
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
            
            # To approximate true transfer entropy, we should subtract I(y_present; y_future)
            # but for ranking features, the raw MI can suffice in this simplified version
    
    return te_scores

# For efficiency, first reduce dimensionality by selecting features with variance above threshold
var_threshold = np.percentile(np.var(X_subsample, axis=0), 50)  # Top 50% by variance
high_var_indices = np.where(np.var(X_subsample, axis=0) > var_threshold)[0]
X_var_reduced = X_subsample[:, high_var_indices]

# Calculate transfer entropy scores on high variance features to improve efficiency
print(f"Calculating transfer entropy on {X_var_reduced.shape[1]} high variance features...")
te_scores_reduced = estimate_transfer_entropy(X_var_reduced, y_subsample[:X_var_reduced.shape[0]])

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

# Random Forest and LASSO with 5-fold cross-validation for each feature selection method
print("\nTraining and evaluating models...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# LASSO classifier for LASSO-selected features
print(f"\nCross-validating LASSO-selected features with LASSO classifier...")
lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
scores_lasso = cross_val_score(lasso_clf, X_lasso_top, y, cv=cv, scoring='accuracy')
print(f"LASSO top {num_features} features (with LASSO classifier) - CV Accuracy: {scores_lasso.mean():.4f} ± {scores_lasso.std():.4f}")

# Random Forest for MI and TE features
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate Mutual Information selected features with Random Forest
print(f"\nCross-validating Mutual Information-selected features with Random Forest...")
scores_mi = cross_val_score(rf, X_mi_top, y, cv=cv, scoring='accuracy')
print(f"MI top {num_features} features (with RF) - CV Accuracy: {scores_mi.mean():.4f} ± {scores_mi.std():.4f}")

# Evaluate Transfer Entropy selected features with Random Forest
print(f"\nCross-validating Transfer Entropy-selected features with Random Forest...")
scores_te = cross_val_score(rf, X_te_top, y, cv=cv, scoring='accuracy')
print(f"TE top {num_features} features (with RF) - CV Accuracy: {scores_te.mean():.4f} ± {scores_te.std():.4f}")

# Create a summary of results
print("\n=== SUMMARY OF FEATURE SELECTION METHODS ===")
results = [
    ("LASSO (with LASSO classifier)", scores_lasso.mean(), scores_lasso.std()),
    ("Mutual Information (with RF)", scores_mi.mean(), scores_mi.std()),
    ("Transfer Entropy (with RF)", scores_te.mean(), scores_te.std())
]

# Sort methods by accuracy
results.sort(key=lambda x: x[1], reverse=True)

# Print rankings
print("\nRanking of feature selection methods by accuracy:")
for i, (method, mean_acc, std_acc) in enumerate(results, 1):
    print(f"{i}. {method}: {mean_acc:.4f} ± {std_acc:.4f}")

print("\nAnalysis complete!")
