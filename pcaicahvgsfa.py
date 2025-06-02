import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.cluster import FeatureAgglomeration
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

# Method 3: Feature selection using ICA
def method_ica(X, y):
    print("\nMethod 3: Feature selection using ICA")
    start_time = time.time()
    
    # Apply ICA to identify important features
    print("Applying ICA to identify important features...")
    ica = FastICA(n_components=30, random_state=42)
    ica.fit(X)
    
    # Identify most important original features based on ICA loadings
    feature_importance = np.zeros(X.shape[1])
    for i in range(30):
        # Sum the absolute loadings across all components
        feature_importance += np.abs(ica.components_[i])
    
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

# Method 4: Feature selection using Feature Agglomeration
def method_feature_agglom(X, y):
    print("\nMethod 4: Feature selection using Feature Agglomeration")
    start_time = time.time()
    
    num_features = 30
    
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
            
        top_indices = np.array(selected_features)
        X_reduced = X[:, top_indices]
        
        # Display pixel positions of top features
        print("\nTop features selected (pixel positions):")
        for i, idx in enumerate(top_indices[:10]):  # Show first 10
            row = idx // 28
            col = idx % 28
            cluster_id = feature_labels[idx]
            print(f"  Feature {i+1}: Pixel ({row}, {col}), Cluster {cluster_id}")
        
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
        
    except Exception as e:
        print(f"Error running Feature Agglomeration: {e}")
        print("Returning empty results.")
        return 0, 0, np.array([]), None

# Method 5: Feature selection using HVGS (Highly Variable Gene Selection)
def method_hvgs(X, y):
    print("\nMethod 5: Feature selection using HVGS (Highly Variable Gene Selection)")
    start_time = time.time()
    
    # Calculate coefficient of variation for each feature
    print("Calculating coefficient of variation for each feature...")
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    
    # To avoid division by zero, add small epsilon to mean
    epsilon = 1e-10
    feature_cv = feature_stds / (feature_means + epsilon)
    
    # Get top 30 features with highest coefficient of variation
    top_indices = np.argsort(-feature_cv)[:30]
    
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

# Compare results
def compare_results(results):
    print("\n--- Results Summary ---")
    print(f"{'Method':<35} {'Accuracy':<10} {'Std Dev':<10}")
    print("-" * 55)
    
    best_acc = 0
    best_method = ""
    
    for method, (acc, std, _, _) in results.items():
        if acc == 0:  # Skip methods that failed
            continue
        print(f"{method:<35} {acc:.4f}     (±{std:.4f})")
        if acc > best_acc:
            best_acc = acc
            best_method = method
    
    print("\n" + "=" * 55)
    print(f"Best method: {best_method} with accuracy {best_acc:.4f}")
    
    # Compare feature sets
    print("\n--- Feature Overlap Analysis ---")
    methods = list(results.keys())
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            
            # Skip if either method failed
            if results[method1][0] == 0 or results[method2][0] == 0:
                continue
                
            indices1 = results[method1][2]
            indices2 = results[method2][2]
            
            common_features = set(indices1.tolist()).intersection(set(indices2.tolist()))
            print(f"{method1} vs {method2}: {len(common_features)} common features")

def main():
    # Load MNIST data
    X, y = load_mnist_data()
    
    # Store results for all methods
    results = {}
    
    # Method 1: Feature selection using PCA without scaling
    results["PCA without scaling"] = method_pca_without_scaling(X, y)
    
    # Method 2: Feature selection using PCA with scaling
    results["PCA with scaling"] = method_pca_with_scaling(X, y)
    
    # Method 3: Feature selection using ICA
    results["ICA"] = method_ica(X, y)
    
    # Method 4: Feature selection using Feature Agglomeration
    results["Feature Agglomeration"] = method_feature_agglom(X, y)
    
    # Method 5: Feature selection using HVGS
    results["HVGS (Highly Variable Gene Selection)"] = method_hvgs(X, y)
    
    # Compare results from all methods
    compare_results(results)
    
    # Additional information about the reduced datasets
    print("\n--- Reduced Dataset Information ---")
    for method, (acc, _, _, X_reduced) in results.items():
        if acc == 0 or X_reduced is None:  # Skip methods that failed
            continue
        print(f"{method} reduced dataset shape: {X_reduced.shape}")

if __name__ == "__main__":
    total_start = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
