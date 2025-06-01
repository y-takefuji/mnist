import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import time

# Load the MNIST data
def load_mnist_data():
    print("Loading MNIST dataset...")
    start_time = time.time()
    
    # Load the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Loading took {time.time() - start_time:.2f} seconds")
    
    return X, y

# Method 1: Feature selection using PCA without scaling
def method_pca_without_scaling(X, y):
    print("\nMethod 1: Feature selection using PCA without scaling")
    start_time = time.time()
    
    # Apply PCA without scaling to identify important features
    print("Applying PCA without scaling to identify important features...")
    pca = PCA(n_components=30)
    pca.fit(X)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    print(f"Explained variance by 30 components: {cumulative_variance[-1]:.4f}")
    
    # Identify most important original features based on PCA loadings
    feature_importance = np.zeros(X.shape[1])
    for i in range(30):
        # Sum the absolute loadings across all components
        feature_importance += np.abs(pca.components_[i])
    
    # Get top 30 features
    top_indices = np.argsort(-feature_importance)[:30]
    
    # Create reduced dataset with selected features
    X_reduced = X[:, top_indices]
    
    # Display pixel positions of top features
    print("\nTop 30 features selected (pixel positions):")
    for i, idx in enumerate(top_indices[:10]):  # Show first 10
        row = idx // 28
        col = idx % 28
        print(f"  Feature {i+1}: Pixel ({row}, {col})")
    
    # Now cross-validate on the reduced dataset
    print("\nPerforming cross-validation on reduced dataset...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_reduced, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print(f"Method execution took {time.time() - start_time:.2f} seconds")
    
    return scores.mean(), scores.std(), top_indices, X_reduced

# Method 2: Feature selection using PCA with scaling
def method_pca_with_scaling(X, y):
    print("\nMethod 2: Feature selection using PCA with scaling")
    start_time = time.time()
    
    # Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to identify important features
    print("Applying PCA to scaled data to identify important features...")
    pca = PCA(n_components=30)
    pca.fit(X_scaled)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    print(f"Explained variance by 30 components: {cumulative_variance[-1]:.4f}")
    
    # Identify most important original features based on PCA loadings
    feature_importance = np.zeros(X.shape[1])
    for i in range(30):
        # Sum the absolute loadings across all components
        feature_importance += np.abs(pca.components_[i])
    
    # Get top 30 features
    top_indices = np.argsort(-feature_importance)[:30]
    
    # Create reduced dataset with selected features
    X_reduced = X[:, top_indices]  # Using original X, not X_scaled
    
    # Display pixel positions of top features
    print("\nTop 30 features selected (pixel positions):")
    for i, idx in enumerate(top_indices[:10]):  # Show first 10
        row = idx // 28
        col = idx % 28
        print(f"  Feature {i+1}: Pixel ({row}, {col})")
    
    # Now cross-validate on the reduced dataset
    print("\nPerforming cross-validation on reduced dataset...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_reduced, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print(f"Method execution took {time.time() - start_time:.2f} seconds")
    
    return scores.mean(), scores.std(), top_indices, X_reduced

# Compare results
def compare_results(unscaled_acc, unscaled_std, scaled_acc, scaled_std, unscaled_indices, scaled_indices):
    # Compare feature sets
    common_features = set(unscaled_indices.tolist()).intersection(set(scaled_indices.tolist()))
    print(f"\nNumber of common features between methods: {len(common_features)}")
    if common_features and len(common_features) <= 10:
        print(f"Common features (pixel positions):")
        for idx in list(common_features)[:10]:
            row = idx // 28
            col = idx % 28
            print(f"  Pixel ({row}, {col})")
    
    print("\n--- Results Summary ---")
    print(f"Method 1 (PCA without scaling) - Accuracy: {unscaled_acc:.4f} (±{unscaled_std:.4f})")
    print(f"Method 2 (PCA with scaling) - Accuracy: {scaled_acc:.4f} (±{scaled_std:.4f})")
    
    if scaled_acc > unscaled_acc:
        print("Method 2 with scaling performed better for this dataset.")
    elif unscaled_acc > scaled_acc:
        print("Method 1 without scaling performed better for this dataset.")
    else:
        print("Both methods performed equally.")

def main():
    # Load MNIST data
    X, y = load_mnist_data()
    
    # Method 1: Feature selection using PCA without scaling
    unscaled_acc, unscaled_std, unscaled_indices, X_reduced_unscaled = method_pca_without_scaling(X, y)
    
    # Method 2: Feature selection using PCA with scaling
    scaled_acc, scaled_std, scaled_indices, X_reduced_scaled = method_pca_with_scaling(X, y)
    
    # Compare results
    compare_results(unscaled_acc, unscaled_std, scaled_acc, scaled_std, unscaled_indices, scaled_indices)
    
    # Additional information about the reduced datasets
    print("\n--- Reduced Dataset Information ---")
    print(f"Method 1 reduced dataset shape: {X_reduced_unscaled.shape}")
    print(f"Method 2 reduced dataset shape: {X_reduced_scaled.shape}")

if __name__ == "__main__":
    total_start = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
