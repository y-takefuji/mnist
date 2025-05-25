# Without scaling + PCA transformation
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data if isinstance(mnist.data, pd.DataFrame) else pd.DataFrame(mnist.data)  # Ensure DataFrame format
y = mnist.target

# Print original dataset shape
print(f"Original dataset shape: {X.shape}")

# Apply PCA to determine feature importance
pca = PCA(n_components=X.shape[1])  # Fit PCA on full dataset
pca.fit(X)

# Compute importance of each feature by summing absolute values of principal components
feature_importance = np.abs(pca.components_).sum(axis=0)

# Sort features by importance and select the top 30
top_features_idx = np.argsort(-feature_importance)[:30]
X_reduced = X.iloc[:, top_features_idx]  # Extract reduced dataset with top 30 features

# Print reduced dataset shape
print(f"Reduced dataset shape: {X_reduced.shape}")

# Perform cross-validation using Random Forest on the reduced dataset
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_classifier, X_reduced, y, cv=5, scoring='accuracy')

# Print results with the requested format
print("\nRandom Forest with PCA-based Feature Selection:")
print(f"Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (5-fold CV)")

print("\nDone!")
