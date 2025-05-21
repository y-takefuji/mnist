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

# First try importing UMAP
try:
    import umap
    umap_available = True
except ImportError:
    print("UMAP not installed. Skipping UMAP feature selection.")
    umap_available = False

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

# NMF-based feature selection
print(f"\nPerforming NMF-based feature selection for top {num_features} features...")
try:
    nmf = NMF(n_components=50, random_state=42, max_iter=500)
    nmf.fit(X)
    
    # Get feature importance from NMF
    # Sum of components for each feature
    nmf_importance = np.sum(nmf.components_, axis=0)
    top_nmf_indices = np.argsort(-nmf_importance)[:num_features]
    X_nmf_top = X[:, top_nmf_indices]
    print(f"NMF top features dataset shape: {X_nmf_top.shape}")
    
    # Display top 5 features from NMF
    print("\nTop 5 NMF most influential original features:")
    for i in range(5):
        feature_idx = top_nmf_indices[i]
        importance = nmf_importance[feature_idx]
        pixel_row = feature_idx // 28
        pixel_col = feature_idx % 28
        print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")
    nmf_available = True
except Exception as e:
    print(f"Error running NMF: {e}")
    print("Continuing without NMF feature selection.")
    nmf_available = False
    X_nmf_top = None

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


# t-SNE based feature selection
print(f"\nPerforming t-SNE based feature selection for top {num_features} features...")
try:
    # Since t-SNE is computationally intensive, we'll use a subset of the data
    sample_size = min(5000, X.shape[0])  # Even smaller sample than UMAP due to computational cost
    np.random.seed(42)
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    
    # Reduce to 2 components with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(X_sample)
    
    # Calculate feature importance based on correlation with t-SNE components
    tsne_feature_importance = np.zeros(X.shape[1])
    for i in range(embedding.shape[1]):
        component = embedding[:, i]
        correlations = np.array([np.abs(np.corrcoef(component, X_sample[:, j])[0, 1]) 
                                 for j in range(X.shape[1])])
        correlations = np.nan_to_num(correlations)
        tsne_feature_importance += correlations
        
    top_tsne_indices = np.argsort(-tsne_feature_importance)[:num_features]
    X_tsne_top = X[:, top_tsne_indices]
    print(f"t-SNE top features dataset shape: {X_tsne_top.shape}")
    
    # Display top 5 features from t-SNE
    print("\nTop 5 t-SNE most influential original features:")
    for i in range(5):
        feature_idx = top_tsne_indices[i]
        importance = tsne_feature_importance[feature_idx]
        pixel_row = feature_idx // 28
        pixel_col = feature_idx % 28
        print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")
    tsne_available = True
except Exception as e:
    print(f"Error running t-SNE: {e}")
    print("Continuing without t-SNE feature selection.")
    tsne_available = False
    X_tsne_top = None

# LLE-based feature selection
print(f"\nPerforming LLE-based feature selection for top {num_features} features...")
try:
    # Use a subset of data for LLE as well
    sample_size = min(5000, X.shape[0]) 
    np.random.seed(42)
    if 'sample_indices' not in locals():
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[sample_indices]
    
    # Apply LLE with 10 neighbors and 10 components
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=10, random_state=42)
    lle_embedding = lle.fit_transform(X_sample)
    
    # Calculate feature importance based on correlation with LLE components
    lle_feature_importance = np.zeros(X.shape[1])
    for i in range(lle_embedding.shape[1]):
        component = lle_embedding[:, i]
        correlations = np.array([np.abs(np.corrcoef(component, X_sample[:, j])[0, 1]) 
                                 for j in range(X.shape[1])])
        correlations = np.nan_to_num(correlations)
        lle_feature_importance += correlations
        
    top_lle_indices = np.argsort(-lle_feature_importance)[:num_features]
    X_lle_top = X[:, top_lle_indices]
    print(f"LLE top features dataset shape: {X_lle_top.shape}")
    
    # Display top 5 features from LLE
    print("\nTop 5 LLE most influential original features:")
    for i in range(5):
        feature_idx = top_lle_indices[i]
        importance = lle_feature_importance[feature_idx]
        pixel_row = feature_idx // 28
        pixel_col = feature_idx % 28
        print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")
    lle_available = True
except Exception as e:
    print(f"Error running LLE: {e}")
    print("Continuing without LLE feature selection.")
    lle_available = False
    X_lle_top = None

# UMAP-based feature selection if available
X_umap_top = None
if umap_available:
    try:
        print(f"\nPerforming UMAP-based feature selection for top {num_features} features...")
        # Since UMAP is computationally intensive, we'll use a subset of the data to fit it
        sample_size = min(10000, X.shape[0])
        np.random.seed(42)
        if 'sample_indices' not in locals() or len(sample_indices) != sample_size:
            sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_indices]

        # Fit UMAP to reduce to 10 dimensions
        reducer = umap.UMAP(n_components=10, random_state=42)
        embedding = reducer.fit_transform(X_sample)

        # Get feature importance from UMAP
        # Calculate correlation between UMAP components and original features
        umap_feature_importance = np.zeros(X.shape[1])
        for i in range(embedding.shape[1]):
            component = embedding[:, i]
            # Calculate correlation between this component and each feature
            correlations = np.array([np.abs(np.corrcoef(component, X_sample[:, j])[0, 1]) for j in range(X.shape[1])])
            # Handle NaNs
            correlations = np.nan_to_num(correlations)
            umap_feature_importance += correlations

        top_umap_indices = np.argsort(-umap_feature_importance)[:num_features]
        X_umap_top = X[:, top_umap_indices]
        print(f"UMAP top features dataset shape: {X_umap_top.shape}")

        # Display top 5 features from UMAP
        print("\nTop 5 UMAP most influential original features:")
        for i in range(5):
            feature_idx = top_umap_indices[i]
            importance = umap_feature_importance[feature_idx]
            pixel_row = feature_idx // 28
            pixel_col = feature_idx % 28
            print(f"Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")
    except Exception as e:
        print(f"Error running UMAP: {e}")
        print("Continuing without UMAP feature selection.")
        umap_available = False
else:
    print("UMAP not available, skipping UMAP feature selection.")

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

# Spearman correlation-based feature selection
print(f"\nPerforming Spearman correlation-based feature selection for top {num_features} features...")
# Calculate Spearman correlation between each feature and the target
spearman_corrs = []
for i in range(X.shape[1]):
    corr, _ = spearmanr(X[:, i], y)
    if np.isnan(corr):  # Handle constant features
        corr = 0
    spearman_corrs.append((i, abs(corr)))

# Select top features based on absolute correlation
spearman_corrs.sort(key=lambda x: x[1], reverse=True)
top_spearman_features = [x[0] for x in spearman_corrs[:num_features]]
X_spearman = X[:, top_spearman_features]
print(f"Spearman-reduced dataset shape: {X_spearman.shape}")

# Display top 5 Spearman features
print("\nTop 5 Spearman features:")
for i in range(5):
    feature_idx = top_spearman_features[i]
    corr_value = spearman_corrs[i][1]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"Feature {feature_idx}: Correlation = {corr_value:.4f}, Position = ({pixel_row}, {pixel_col})")

# Kendall tau-based feature selection
print(f"\nPerforming Kendall tau-based feature selection for top {num_features} features...")
# Calculate Kendall tau correlation between each feature and the target
kendall_corrs = []
for i in range(X.shape[1]):
    corr, _ = kendalltau(X[:, i], y)
    if np.isnan(corr):  # Handle constant features
        corr = 0
    kendall_corrs.append((i, abs(corr)))

# Select top features based on absolute correlation
kendall_corrs.sort(key=lambda x: x[1], reverse=True)
top_kendall_features = [x[0] for x in kendall_corrs[:num_features]]
X_kendall = X[:, top_kendall_features]
print(f"Kendall-reduced dataset shape: {X_kendall.shape}")

# Display top 5 Kendall features
print("\nTop 5 Kendall tau features:")
for i in range(5):
    feature_idx = top_kendall_features[i]
    corr_value = kendall_corrs[i][1]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"Feature {feature_idx}: Correlation = {corr_value:.4f}, Position = ({pixel_row}, {pixel_col})")

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

# Evaluate with NMF top features if available
if nmf_available and X_nmf_top is not None:
    scores_nmf = cross_val_score(rf, X_nmf_top, y, cv=cv, scoring='accuracy')
    print(f"NMF top {num_features} features - CV Accuracy: {scores_nmf.mean():.4f} ± {scores_nmf.std():.4f}")

# Evaluate with Feature Agglomeration if available
if agglo_available and X_agglo is not None:
    scores_agglo = cross_val_score(rf, X_agglo, y, cv=cv, scoring='accuracy')
    print(f"Feature Agglomeration {num_features} features - CV Accuracy: {scores_agglo.mean():.4f} ± {scores_agglo.std():.4f}")

# Evaluate with t-SNE top features if available
if tsne_available and X_tsne_top is not None:
    scores_tsne = cross_val_score(rf, X_tsne_top, y, cv=cv, scoring='accuracy')
    print(f"t-SNE top {num_features} features - CV Accuracy: {scores_tsne.mean():.4f} ± {scores_tsne.std():.4f}")

# Evaluate with LLE top features if available
if lle_available and X_lle_top is not None:
    scores_lle = cross_val_score(rf, X_lle_top, y, cv=cv, scoring='accuracy')
    print(f"LLE top {num_features} features - CV Accuracy: {scores_lle.mean():.4f} ± {scores_lle.std():.4f}")

# Evaluate with UMAP top features if available
if umap_available and X_umap_top is not None:
    scores_umap = cross_val_score(rf, X_umap_top, y, cv=cv, scoring='accuracy')
    print(f"UMAP top {num_features} features - CV Accuracy: {scores_umap.mean():.4f} ± {scores_umap.std():.4f}")

# Evaluate with HVGS features
scores_hvgs = cross_val_score(rf, X_hvgs, y, cv=cv, scoring='accuracy')
print(f"HVGS top {num_features} features - CV Accuracy: {scores_hvgs.mean():.4f} ± {scores_hvgs.std():.4f}")

# Evaluate with Spearman correlation features
scores_spearman = cross_val_score(rf, X_spearman, y, cv=cv, scoring='accuracy')
print(f"Spearman top {num_features} features - CV Accuracy: {scores_spearman.mean():.4f} ± {scores_spearman.std():.4f}")

# Evaluate with Kendall tau features
scores_kendall = cross_val_score(rf, X_kendall, y, cv=cv, scoring='accuracy')
print(f"Kendall tau top {num_features} features - CV Accuracy: {scores_kendall.mean():.4f} ± {scores_kendall.std():.4f}")

# Create a summary of results
print("\n=== SUMMARY OF FEATURE SELECTION METHODS ===")
results = [
    ("PCA", scores_pca.mean(), scores_pca.std()),
    ("ICA", scores_ica.mean(), scores_ica.std()),
    ("HVGS", scores_hvgs.mean(), scores_hvgs.std()),
    ("Spearman", scores_spearman.mean(), scores_spearman.std()),
    ("Kendall", scores_kendall.mean(), scores_kendall.std())
]

if nmf_available and X_nmf_top is not None:
    results.append(("NMF", scores_nmf.mean(), scores_nmf.std()))
if agglo_available and X_agglo is not None:
    results.append(("Feature Agglomeration", scores_agglo.mean(), scores_agglo.std()))
if tsne_available and X_tsne_top is not None:
    results.append(("t-SNE", scores_tsne.mean(), scores_tsne.std()))
if lle_available and X_lle_top is not None:
    results.append(("LLE", scores_lle.mean(), scores_lle.std()))
if umap_available and X_umap_top is not None:
    results.append(("UMAP", scores_umap.mean(), scores_umap.std()))

# Sort methods by accuracy
results.sort(key=lambda x: x[1], reverse=True)

# Print rankings
print("\nRanking of feature selection methods by accuracy:")
for i, (method, mean_acc, std_acc) in enumerate(results, 1):
    print(f"{i}. {method}: {mean_acc:.4f} ± {std_acc:.4f}")

print("\nAnalysis complete!")
