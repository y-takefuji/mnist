import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
import time

# Set random seed for reproducibility
np.random.seed(42)

# Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Convert pandas DataFrame to numpy array to avoid indexing issues
if isinstance(X, pd.DataFrame):
    X = X.values
if isinstance(y, pd.Series):
    y = y.values

print(f"MNIST dataset shape: {X.shape}")

# Define function to select highly variable genes/features
def select_hvgs(X, n_features=30):
    """Select the top n_features with highest variance"""
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[::-1][:n_features]
    return X[:, top_idx], top_idx

# Function to evaluate a feature selection method with cross-validation
def evaluate_feature_selection(name, X_reduced, y):
    print(f"\nEvaluating {name} (shape: {X_reduced.shape})...")
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Perform 5-fold cross-validation
    start_time = time.time()
    scores = cross_val_score(rf, X_reduced, y, cv=5, scoring='accuracy')
    elapsed_time = time.time() - start_time
    
    print(f"{name} - Cross-validation scores: {scores}")
    print(f"{name} - Mean accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
    print(f"{name} - Cross-validation time: {elapsed_time:.2f} seconds")
    
    return scores.mean(), scores.std()

# Define number of features to select
n_features = 30

# 1. Feature Agglomeration
print("\nPerforming Feature Agglomeration...")
feature_agglomeration = FeatureAgglomeration(n_clusters=n_features)
feature_agglomeration.fit(X)
# For each cluster, select the feature closest to the centroid
X_fa = np.zeros((X.shape[0], n_features))
for i in range(n_features):
    cluster_indices = np.where(feature_agglomeration.labels_ == i)[0]
    # Select the first feature from each cluster (simplification)
    if len(cluster_indices) > 0:
        X_fa[:, i] = X[:, cluster_indices[0]]
print(f"Feature Agglomeration dataset shape: {X_fa.shape}")

# 2. PCA - selecting original features based on importance
print("\nPerforming PCA...")
pca = PCA(n_components=n_features)
_ = pca.fit(X)  # Only fit to get components
# Get the most important original feature for each principal component
important_features = []
for i in range(n_features):
    most_important = np.abs(pca.components_[i]).argmax()
    important_features.append(most_important)
X_pca = X[:, important_features]
print(f"PCA selected original features dataset shape: {X_pca.shape}")

# 3. HVGs (Highly Variable Features)
print("\nSelecting Highly Variable Features...")
X_hvg, hvg_indices = select_hvgs(X, n_features=n_features)
print(f"HVGs dataset shape: {X_hvg.shape}")
print(f"Selected feature indices with highest variance: {hvg_indices[:5]}...")

# Evaluate each method with cross-validation
results = []
print("\nCross-validating with Random Forest on each reduced dataset:")
results.append(evaluate_feature_selection("Feature Agglomeration", X_fa, y))
results.append(evaluate_feature_selection("PCA", X_pca, y))
results.append(evaluate_feature_selection("HVGs", X_hvg, y))

# Print summary of results
print("\n----- SUMMARY OF RESULTS -----")
methods = ["Feature Agglomeration", "PCA", "HVGs"]
for i, method in enumerate(methods):
    print(f"{method}: {results[i][0]:.4f} (±{results[i][1]:.4f})")

print("\nAnalysis complete.")
