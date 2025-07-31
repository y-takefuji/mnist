#  Affinity Propagation
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time

# Set random seed for reproducibility
np.random.seed(42)

# Load MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")

# Convert to numpy arrays
X = X.to_numpy().astype(np.float32)
y = y.to_numpy().astype(np.int64)

# Use a subset of 70,000 samples as specified
X = X[:70000]
y = y[:70000]

# Compute feature importance using variance
print("Computing feature variances...")
feature_variance = np.var(X, axis=0)

# Select 100 highest variance features to reduce computational load for Affinity Propagation
top_variance_indices = np.argsort(feature_variance)[-100:]
X_reduced = X[:, top_variance_indices]

# Create a correlation matrix between features
print("Computing feature correlation matrix...")
# Transpose so that each row represents a feature
corr_matrix = np.corrcoef(X_reduced.T)

# Apply Affinity Propagation to select exemplar features
print("Applying Affinity Propagation for feature selection...")
start_time = time.time()

# Try different preference values until we get close to 30 clusters
preference = -50
max_attempts = 10
attempt = 0

while attempt < max_attempts:
    ap = AffinityPropagation(random_state=42, max_iter=200, convergence_iter=15, preference=preference)
    ap.fit(corr_matrix)
    cluster_centers_indices = ap.cluster_centers_indices_
    num_clusters = len(cluster_centers_indices)
    
    print(f"Attempt {attempt+1}: preference={preference}, found {num_clusters} clusters")
    
    if num_clusters == 30:
        break
    elif num_clusters < 30:
        preference += 10  # Increase preference to get more clusters
    else:  # num_clusters > 30
        # If we have more than 30, we can just take 30 of them
        break
    
    attempt += 1

ap_time = time.time() - start_time
print(f"Affinity Propagation completed in {ap_time:.2f} seconds")
print(f"Number of clusters found: {len(cluster_centers_indices)}")

# Get the final 30 features
if len(cluster_centers_indices) >= 30:
    # If we have more than 30 clusters, select based on cluster size
    cluster_sizes = np.bincount(ap.labels_)
    # Get the indices of the 30 largest clusters
    top_clusters_indices = np.argsort(cluster_sizes)[-30:]
    # Get the centers of these clusters
    selected_feature_indices = [cluster_centers_indices[i] for i in top_clusters_indices]
    # Ensure we have exactly 30
    selected_feature_indices = selected_feature_indices[:30]
else:
    # If we have fewer than 30, take all cluster centers and add more features
    # based on variance until we have 30
    missing = 30 - len(cluster_centers_indices)
    # Start with all cluster centers
    selected_feature_indices = list(cluster_centers_indices)
    
    # Find which features are already selected
    all_features = set(range(X_reduced.shape[1]))
    selected_set = set(selected_feature_indices)
    
    # Get remaining features sorted by variance
    remaining_features = sorted(list(all_features - selected_set), 
                              key=lambda i: feature_variance[top_variance_indices[i]], 
                              reverse=True)
    
    # Add the top remaining features by variance
    selected_feature_indices.extend(remaining_features[:missing])

# Map back to original feature indices
final_feature_indices = [top_variance_indices[i] for i in selected_feature_indices]
print(f"Selected {len(final_feature_indices)} features: {final_feature_indices}")

# Create dataset with selected features
X_selected = X[:, final_feature_indices]
print(f"Reduced dataset shape: {X_selected.shape}")

# Cross-validate with Random Forest
print("Cross-validating with Random Forest classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
scores = cross_val_score(rf, X_selected, y, cv=5, scoring='accuracy', n_jobs=-1)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Compare with using all features (on a smaller subset due to computational constraints)
sample_size = 10000  # Use a smaller sample for full feature evaluation
X_sample = X[:sample_size]
y_sample = y[:sample_size]
X_selected_sample = X_selected[:sample_size]

print(f"\nComparing with full feature set on {sample_size} samples...")
rf_full = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
scores_full = cross_val_score(rf_full, X_sample, y_sample, cv=3, scoring='accuracy', n_jobs=-1)

print(f"Full feature set ({X.shape[1]} features) - Mean accuracy: {scores_full.mean():.4f} ± {scores_full.std():.4f}")

rf_selected = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
scores_selected = cross_val_score(rf_selected, X_selected_sample, y_sample, cv=3, scoring='accuracy', n_jobs=-1)

print(f"Selected features ({X_selected.shape[1]} features) - Mean accuracy: {scores_selected.mean():.4f} ± {scores_selected.std():.4f}")
