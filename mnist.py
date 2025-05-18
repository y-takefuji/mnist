import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import spearmanr, kendalltau
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.cluster import FeatureAgglomeration
from sklearn.random_projection import SparseRandomProjection


# Load NIST handwritten digits (MNIST) dataset
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

# Number of features to select
num_features = 30

# PCA-based feature selection
print(f"\nPerforming PCA-based feature selection for top {num_features} features...")
pca = PCA(n_components=50)  # We still fit 50 components to capture enough variance
pca.fit(X)

# Get feature importance from PCA
# Sum of squared loadings across all components, weighted by explained variance
feature_importance = np.sum(np.square(pca.components_) * pca.explained_variance_ratio_.reshape(-1, 1), axis=0)
top_pca_indices = np.argsort(-feature_importance)[:num_features]  # Get indices of top features
X_pca_top = X[:, top_pca_indices]
print(f"PCA top features dataset shape: {X_pca_top.shape}")

# Display top 5 features from PCA
print("\nTop 5 PCA most influential original features:")
for i in range(5):
    feature_idx = top_pca_indices[i]
    importance = feature_importance[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")

# ICA-based feature selection
print(f"\nPerforming ICA-based feature selection for top {num_features} features...")
ica = FastICA(n_components=50, random_state=42, max_iter=500)
ica.fit(X)

# Get feature importance from ICA
# Sum of absolute values of ICA components for each feature
ica_importance = np.sum(np.abs(ica.components_), axis=0)
top_ica_indices = np.argsort(-ica_importance)[:num_features]
X_ica_top = X[:, top_ica_indices]
print(f"ICA top features dataset shape: {X_ica_top.shape}")

# Display top 5 features from ICA
print("\nTop 5 ICA most influential original features:")
for i in range(5):
    feature_idx = top_ica_indices[i]
    importance = ica_importance[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")


# Feature Agglomeration for feature selection
print(f"\nPerforming Feature Agglomeration for feature selection...")
try:
    # Feature Agglomeration - cluster features into groups
    n_clusters = num_features
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    X_reduced = agglo.fit_transform(X)
    
    # For feature selection we need to determine the most representative features
    # From each cluster, we'll select the feature closest to the cluster center
    X_transformed = agglo.fit_transform(X)
    
    # Get the labels for each feature
    feature_labels = agglo.labels_
    
    # For each cluster, find the feature closest to the cluster center
    selected_features = []
    for cluster_id in range(n_clusters):
        # Find features in this cluster
        cluster_features = np.where(feature_labels == cluster_id)[0]
        
        if len(cluster_features) == 0:
            continue
            
        # If there's only one feature in the cluster, select it
        if len(cluster_features) == 1:
            selected_features.append(cluster_features[0])
            continue
            
        # Otherwise, find the feature closest to the cluster center
        # For each feature in the cluster, calculate the sum of distances to other features
        min_dist_sum = float('inf')
        representative_feature = -1
        
        for i, feat_i in enumerate(cluster_features):
            dist_sum = 0
            for j, feat_j in enumerate(cluster_features):
                if i != j:
                    # Calculate distance between features using correlation
                    corr = np.corrcoef(X[:, feat_i], X[:, feat_j])[0, 1]
                    # Convert correlation to distance (1 - |correlation|)
                    dist = 1 - abs(corr)
                    dist_sum += dist
            
            if dist_sum < min_dist_sum:
                min_dist_sum = dist_sum
                representative_feature = feat_i
        
        if representative_feature != -1:
            selected_features.append(representative_feature)
    
    # If we don't have enough features, add more until we reach num_features
    # Choose remaining features with highest variance
    if len(selected_features) < num_features:
        # Calculate variance of each feature
        variances = np.var(X, axis=0)
        # Sort features by variance
        sorted_indices = np.argsort(-variances)
        # Add features not already selected
        for idx in sorted_indices:
            if idx not in selected_features:
                selected_features.append(idx)
                if len(selected_features) >= num_features:
                    break
    
    # Select only the top num_features if we have more
    if len(selected_features) > num_features:
        selected_features = selected_features[:num_features]
        
    X_agglo = X[:, selected_features]
    print(f"Feature Agglomeration dataset shape: {X_agglo.shape}")
    
    # Display top 5 features from Feature Agglomeration
    print("\nTop 5 Feature Agglomeration selected features:")
    for i in range(min(5, len(selected_features))):
        feature_idx = selected_features[i]
        cluster_id = feature_labels[feature_idx]
        pixel_row = feature_idx // 28
        pixel_col = feature_idx % 28
        print(f"Feature {feature_idx}: Cluster {cluster_id}, Position = ({pixel_row}, {pixel_col})")
    
    agglo_available = True
    
except Exception as e:
    print(f"Error running Feature Agglomeration: {e}")
    print("Continuing without Feature Agglomeration feature selection.")
    agglo_available = False
    X_agglo = None

# HVGS (High Variance Gene Selection) feature selection
print(f"\nPerforming HVGS feature selection for top {num_features} features...")

# Calculate variance of each feature
feature_variances = np.var(X, axis=0)

# Sort features by variance
hvgs_indices = np.argsort(-feature_variances)[:num_features]
X_hvgs = X[:, hvgs_indices]
print(f"HVGS-reduced dataset shape: {X_hvgs.shape}")

# Display top 5 HVGS features
print("\nTop 5 HVGS features:")
for i in range(5):
    feature_idx = hvgs_indices[i]
    variance = feature_variances[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"Feature {feature_idx}: Variance = {variance:.4f}, Position = ({pixel_row}, {pixel_col})")


# Random Forest with 5-fold cross-validation for each feature selection method
print("\nTraining and evaluating Random Forest models...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate with PCA top features
scores_pca = cross_val_score(rf, X_pca_top, y, cv=cv, scoring='accuracy')
print(f"PCA top {num_features} features - CV Accuracy: {scores_pca.mean():.4f} ± {scores_pca.std():.4f}")

# Evaluate with ICA top features
scores_ica = cross_val_score(rf, X_ica_top, y, cv=cv, scoring='accuracy')
print(f"ICA top {num_features} features - CV Accuracy: {scores_ica.mean():.4f} ± {scores_ica.std():.4f}")


# Evaluate with Feature Agglomeration if available
if agglo_available and X_agglo is not None:
    scores_agglo = cross_val_score(rf, X_agglo, y, cv=cv, scoring='accuracy')
    print(f"Feature Agglomeration {num_features} features - CV Accuracy: {scores_agglo.mean():.4f} ± {scores_agglo.std():.4f}")

# Evaluate with HVGS features
scores_hvgs = cross_val_score(rf, X_hvgs, y, cv=cv, scoring='accuracy')
print(f"HVGS top {num_features} features - CV Accuracy: {scores_hvgs.mean():.4f} ± {scores_hvgs.std():.4f}")

# Create a summary of results
print("\n=== SUMMARY OF FEATURE SELECTION METHODS ===")
results = [
    ("PCA", scores_pca.mean(), scores_pca.std()),
    ("ICA", scores_ica.mean(), scores_ica.std()),
    ("HVGS", scores_hvgs.mean(), scores_hvgs.std())
    ]

if agglo_available and X_agglo is not None:
    results.append(("Feature Agglomeration", scores_agglo.mean(), scores_agglo.std()))

# Sort methods by accuracy
results.sort(key=lambda x: x[1], reverse=True)

# Print rankings
print("\nRanking of feature selection methods by accuracy:")
for i, (method, mean_acc, std_acc) in enumerate(results, 1):
    print(f"{i}. {method}: {mean_acc:.4f} ± {std_acc:.4f}")

print("\nAnalysis complete!")
