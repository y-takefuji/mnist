# With feature agglomeration for feature selection
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data if isinstance(mnist.data, pd.DataFrame) else pd.DataFrame(mnist.data)  # Ensure DataFrame format
y = mnist.target

# Print original dataset shape
print(f"Original dataset shape: {X.shape}")

# Method 1: Using PCA for feature importance
print("\n--- Method 1: PCA-based Feature Selection ---")
pca = PCA(n_components=X.shape[1])  # Fit PCA on full dataset
pca.fit(X)

# Compute importance of each feature by summing absolute values of principal components
feature_importance = np.abs(pca.components_).sum(axis=0)

# Sort features by importance and select the top 30
top_features_idx = np.argsort(-feature_importance)[:30]
X_reduced_pca = X.iloc[:, top_features_idx]  # Extract reduced dataset with top 30 features

# Print reduced dataset shape
print(f"PCA-reduced dataset shape: {X_reduced_pca.shape}")

# Method 2: Using Feature Agglomeration
print("\n--- Method 2: Feature Agglomeration ---")
# Feature Agglomeration groups similar features together
agglomeration = FeatureAgglomeration(n_clusters=30)
X_reduced_agg = agglomeration.fit_transform(X)

# Print reduced dataset shape
print(f"Agglomeration-reduced dataset shape: {X_reduced_agg.shape}")

# Perform cross-validation using Random Forest on both reduced datasets
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate PCA-based feature selection
print("\nRandom Forest with PCA-based Feature Selection:")
cv_scores_pca = cross_val_score(rf_classifier, X_reduced_pca, y, cv=5, scoring='accuracy')
print(f"Accuracy: {cv_scores_pca.mean():.4f} ± {cv_scores_pca.std():.4f} (5-fold CV)")

# Evaluate Feature Agglomeration
print("\nRandom Forest with Feature Agglomeration:")
cv_scores_agg = cross_val_score(rf_classifier, X_reduced_agg, y, cv=5, scoring='accuracy')
print(f"Accuracy: {cv_scores_agg.mean():.4f} ± {cv_scores_agg.std():.4f} (5-fold CV)")

print("\nDone!")
