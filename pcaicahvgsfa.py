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

# High Variance Gene Selection function
def select_hvgs(X, n_features=30):
    """Select the top n_features with highest variance"""
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[::-1][:n_features]
    return X[:, top_idx], top_idx

# Method 1: PCA without scaling (NO TRANSFORM - feature selection only)
def method_pca_without_scaling(X, y):
    print("\nMethod 1: PCA without scaling (NO TRANSFORM - feature selection only)")
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

# Method 2: PCA with scaling (WITH TRANSFORM)
def method_pca_with_scaling(X, y):
    print("\nMethod 2: PCA with scaling (WITH TRANSFORM)")
    start_time = time.time()
    
    # Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to transform data
    print("Applying PCA to scaled data to transform data...")
    pca = PCA(n_components=30)
    X_transformed = pca.fit_transform(X_scaled)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    print(f"Explained variance by 30 components: {cumulative_variance[-1]:.4f}")
    
    # Display information about the components
    print("\nPCA component information:")
    for i in range(5):  # Show info for first 5 components
        print(f"  Component {i+1}: Explains {explained_variance[i]:.4f} of variance")
    
    # Now cross-validate on the transformed dataset
    print("\nPerforming cross-validation on transformed dataset...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_transformed, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print(f"Method execution took {time.time() - start_time:.2f} seconds")
    
    return scores.mean(), scores.std(), None, X_transformed

# Method 3: Feature selection using ICA
def method_ica(X, y):
    print("\nMethod 3: Feature selection using ICA")
    start_time = time.time()
    
    num_features = 30
    
    # Apply ICA to identify important features
    print("Applying ICA to identify important features...")
    ica = FastICA(n_components=30, random_state=42)
    ica.fit(X)
    
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
    
    # Show all top features
    print("\nTop 30 features selected (pixel positions):")
    for i, idx in enumerate(top_ica_indices[:10]):  # Show first 10
        row = idx // 28
        col = idx % 28
        importance = ica_importance[idx]
        print(f"  Feature {i+1}: Pixel ({row}, {col}), Importance = {importance:.4f}")
    
    # Now cross-validate on the reduced dataset
    print("\nPerforming cross-validation on reduced dataset...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_ica_top, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print(f"Method execution took {time.time() - start_time:.2f} seconds")
    
    return scores.mean(), scores.std(), top_ica_indices, X_ica_top

# Method 4: Feature selection using Feature Agglomeration
def method_feature_agglom(X, y):
    print("\nMethod 4: Feature selection using Feature Agglomeration")
    start_time = time.time()
    
    num_features = 30
    
    try:
        # Feature Agglomeration - cluster features into groups
        n_clusters = num_features
        agglo = FeatureAgglomeration(n_clusters=n_clusters)
        agglo.fit(X)
        
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

# Method 5: Feature selection using HVGS (High Variance Gene Selection)
def method_hvgs(X, y):
    print("\nMethod 5: Feature selection using HVGS (High Variance Gene Selection)")
    start_time = time.time()
    
    # Apply the HVGS function to select features with highest variance
    print("Selecting features with highest variance...")
    X_reduced, top_indices = select_hvgs(X, n_features=30)
    
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
        print(f"{method:<35} {acc:.4f} ±{std:.4f}")
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
            
            # Skip if either method failed or doesn't have feature indices
            if results[method1][0] == 0 or results[method2][0] == 0:
                continue
                
            indices1 = results[method1][2]
            indices2 = results[method2][2]
            
            # Skip comparison if either method doesn't have indices (e.g., PCA transform)
            if indices1 is None or indices2 is None:
                continue
                
            common_features = set(indices1.tolist()).intersection(set(indices2.tolist()))
            print(f"{method1} vs {method2}: {len(common_features)} common features")

def main():
    # Load MNIST data
    X, y = load_mnist_data()
    
    # Store results for all methods
    results = {}
    
    # Method 1: PCA without scaling (NO TRANSFORM - feature selection only)
    results["PCA without scaling (NO TRANSFORM)"] = method_pca_without_scaling(X, y)
    
    # Method 2: PCA with scaling (WITH TRANSFORM)
    results["PCA with scaling (transform)"] = method_pca_with_scaling(X, y)
    
    # Method 3: Feature selection using ICA
    results["ICA"] = method_ica(X, y)
    
    # Method 4: Feature selection using Feature Agglomeration
    results["Feature Agglomeration"] = method_feature_agglom(X, y)
    
    # Method 5: Feature selection using HVGS
    results["HVGS (High Variance Gene Selection)"] = method_hvgs(X, y)
    
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
