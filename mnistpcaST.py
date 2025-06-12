import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time

# Set random seed for reproducibility
np.random.seed(42)

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32').values  # Convert to numpy array for easier indexing
y = mnist.target.astype('int')

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# METHOD 1: No scaling, No transform - Select top 30 features
print("\n1. METHOD 1: No scaling, No transform - Select top 30 features")
feature_variances = np.var(X, axis=0)
top_30_indices = np.argsort(-feature_variances)[:30]  # Indices of top 30 features by variance
X_method1 = X[:, top_30_indices]  # Select only those features

# METHOD 2: Scaling and Transform (PCA with scaling)
print("2. METHOD 2: Scaling and Transform (PCA with scaling)")
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X)
pca_scaled = PCA(n_components=30)
X_method2 = pca_scaled.fit_transform(X_scaled_full)
feature_importance_method2 = pca_scaled.explained_variance_ratio_

# METHOD 3: No scaling and Transform (PCA without scaling)
print("3. METHOD 3: No scaling and Transform (PCA without scaling)")
pca_unscaled = PCA(n_components=30)
X_method3 = pca_unscaled.fit_transform(X)  # Direct PCA on original data without scaling
feature_importance_method3 = pca_unscaled.explained_variance_ratio_

# METHOD 4: Scaling and No transform - Same positions as Method 1
print("4. METHOD 4: Scaling and No transform - Same positions as Method 1")
X_method4 = X_scaled_full[:, top_30_indices]  # Use same positions but scaled values

# Print top feature information
print("\nTop 5 pixel positions selected for Methods 1 & 4 (by variance):")
for i in range(5):
    idx = top_30_indices[i]
    print(f"Pixel position {idx}: Original variance = {feature_variances[idx]:.4f}")
    # Calculate row and column in the 28x28 image
    row, col = idx // 28, idx % 28
    print(f"  (corresponds to row={row}, column={col} in the image)")

print("\nTop 5 principal components (Method 2 - PCA with scaling):")
for i in range(5):
    print(f"PC {i+1}: Explained Variance Ratio = {feature_importance_method2[i]:.4f}")

print("\nTop 5 principal components (Method 3 - PCA without scaling):")
for i in range(5):
    print(f"PC {i+1}: Explained Variance Ratio = {feature_importance_method3[i]:.4f}")

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# METHOD 1: Cross-validation with original unscaled features
print("\nEvaluating Method 1: No scaling, No transform...")
start_time = time.time()
scores_method1 = cross_val_score(clf, X_method1, y, cv=cv, scoring='accuracy')
end_time = time.time()
duration_method1 = end_time - start_time

print(f"\nMethod 1 (No scaling, No transform) Results:")
print(f"Cross-validation scores: {scores_method1}")
print(f"Mean accuracy: {scores_method1.mean():.4f}")
print(f"Standard deviation: {scores_method1.std():.4f}")
print(f"Time taken: {duration_method1:.2f} seconds")

# METHOD 2: Cross-validation with PCA on scaled data
print("\nEvaluating Method 2: Scaling and Transform (PCA with scaling)...")
start_time = time.time()
scores_method2 = cross_val_score(clf, X_method2, y, cv=cv, scoring='accuracy')
end_time = time.time()
duration_method2 = end_time - start_time

print(f"\nMethod 2 (Scaling and Transform) Results:")
print(f"Cross-validation scores: {scores_method2}")
print(f"Mean accuracy: {scores_method2.mean():.4f}")
print(f"Standard deviation: {scores_method2.std():.4f}")
print(f"Time taken: {duration_method2:.2f} seconds")

# METHOD 3: Cross-validation with PCA features (no scaling)
print("\nEvaluating Method 3: No scaling and Transform (PCA without scaling)...")
start_time = time.time()
scores_method3 = cross_val_score(clf, X_method3, y, cv=cv, scoring='accuracy')
end_time = time.time()
duration_method3 = end_time - start_time

print(f"\nMethod 3 (No scaling and Transform) Results:")
print(f"Cross-validation scores: {scores_method3}")
print(f"Mean accuracy: {scores_method3.mean():.4f}")
print(f"Standard deviation: {scores_method3.std():.4f}")
print(f"Time taken: {duration_method3:.2f} seconds")

# METHOD 4: Cross-validation with scaled features at same positions
print("\nEvaluating Method 4: Scaling and No transform...")
start_time = time.time()
scores_method4 = cross_val_score(clf, X_method4, y, cv=cv, scoring='accuracy')
end_time = time.time()
duration_method4 = end_time - start_time

print(f"\nMethod 4 (Scaling and No transform) Results:")
print(f"Cross-validation scores: {scores_method4}")
print(f"Mean accuracy: {scores_method4.mean():.4f}")
print(f"Standard deviation: {scores_method4.std():.4f}")
print(f"Time taken: {duration_method4:.2f} seconds")

# Compare results of all four methods with the requested format
print("\nComparison of Results (accuracy±std):")
print(f"Method 1 (No scaling, No transform): {scores_method1.mean():.4f}±{scores_method1.std():.4f}")
print(f"Method 2 (Scaling and Transform): {scores_method2.mean():.4f}±{scores_method2.std():.4f}")
print(f"Method 3 (No scaling and Transform): {scores_method3.mean():.4f}±{scores_method3.std():.4f}")
print(f"Method 4 (Scaling and No transform): {scores_method4.mean():.4f}±{scores_method4.std():.4f}")

# Calculate total explained variance for PCA methods
total_var_method2 = np.sum(feature_importance_method2)
total_var_method3 = np.sum(feature_importance_method3)
print(f"\nTotal explained variance (PCA with scaling): {total_var_method2:.4f}")
print(f"Total explained variance (PCA without scaling): {total_var_method3:.4f}")
