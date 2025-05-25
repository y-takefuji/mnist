# With scaling + PCA transformation
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.values if hasattr(mnist.data, 'values') else mnist.data
y = mnist.target
print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")

# Standardize the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to identify top 30 features
print("Applying PCA to reduce to 30 features...")
pca = PCA(n_components=30)
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced dataset shape: {X_reduced.shape}")

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_.sum()
print(f"Explained variance by top 30 principal components: {explained_variance:.4f} ({explained_variance*100:.2f}%)")

# Perform 5-fold cross-validation with Random Forest
print("\nPerforming 5-fold cross-validation with Random Forest...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_classifier, X_reduced, y, cv=5, scoring='accuracy')

# Print results with the requested format
print("\nRandom Forest with PCA Transformation:")
print(f"Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (5-fold CV)")

print("\nDone!")
