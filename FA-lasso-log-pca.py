import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import accuracy_score
import time

# User parameter: Number of top features to select
num_features = 30

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Preprocess dataset
print("Preprocessing dataset...")
X = np.array(X)
y = np.array(y)
print(f"Original MNIST dataset shape: {X.shape}")

# Normalize pixel values
X = X / 255.0

# 1. LASSO-based feature selection
print(f"\nPerforming LASSO-based feature selection for top {num_features} features...")
start_time = time.time()

# For LASSO feature selection, we use a simpler approach for multi-class
# Create a binary target vector: 1 if digit > 4, 0 otherwise (dividing digits into two groups)
binary_y = (y > 4).astype(int)

# Use Lasso to identify important features for this binary task - no scaling
lasso_selector = Lasso(alpha=0.01, max_iter=2000, random_state=42)
lasso_selector.fit(X, binary_y)  # Using X directly without scaling

# Get feature importance from LASSO coefficients
lasso_importance = np.abs(lasso_selector.coef_)
top_lasso_indices = np.argsort(-lasso_importance)[:num_features]
X_lasso_reduced = X[:, top_lasso_indices]
print(f"LASSO top features dataset shape: {X_lasso_reduced.shape}")
print(f"LASSO feature selection completed in {time.time() - start_time:.2f} seconds")

# Display top 5 features from LASSO
print(f"Top 5 LASSO most influential features:")
for i in range(min(5, num_features)):
    feature_idx = top_lasso_indices[i]
    importance = lasso_importance[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"  Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")

# 2. Logistic Regression for feature selection
print(f"\nPerforming Logistic Regression-based feature selection for top {num_features} features...")
start_time = time.time()
log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42, 
                             max_iter=1000, multi_class='ovr')
log_reg.fit(X, y)  # Using X directly without scaling

# Get feature importance from Logistic Regression coefficients
log_reg_importance = np.max(np.abs(log_reg.coef_), axis=0)
top_logreg_indices = np.argsort(-log_reg_importance)[:num_features]
X_logreg_reduced = X[:, top_logreg_indices]
print(f"LogReg top features dataset shape: {X_logreg_reduced.shape}")
print(f"LogReg feature selection completed in {time.time() - start_time:.2f} seconds")

# Display top 5 features from Logistic Regression
print(f"Top 5 LogReg most influential features:")
for i in range(min(5, num_features)):
    feature_idx = top_logreg_indices[i]
    importance = log_reg_importance[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"  Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")

# 3. PCA-based feature selection
print(f"\nPerforming PCA to identify top {num_features} features...")
start_time = time.time()

# For PCA, we'll still use scaled data as it's sensitive to feature scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=num_features, random_state=42)
pca.fit(X_scaled)

# For PCA, identify the most important original features
pca_importance = np.zeros(X.shape[1])
for i in range(num_features):
    # Sum the absolute loadings across all components
    pca_importance += np.abs(pca.components_[i])

top_pca_indices = np.argsort(-pca_importance)[:num_features]
X_pca_reduced = X[:, top_pca_indices]  # Using original X for the final features
print(f"PCA top features dataset shape: {X_pca_reduced.shape}")
print(f"PCA feature selection completed in {time.time() - start_time:.2f} seconds")

# Display top 5 features from PCA
print(f"Top 5 PCA most influential features:")
for i in range(min(5, num_features)):
    feature_idx = top_pca_indices[i]
    importance = pca_importance[feature_idx]
    pixel_row = feature_idx // 28
    pixel_col = feature_idx % 28
    print(f"  Feature {feature_idx}: Importance = {importance:.4f}, Position = ({pixel_row}, {pixel_col})")

# 4. Feature Agglomeration for feature selection
print(f"\nPerforming Feature Agglomeration for feature selection...")
agglo_available = True
start_time = time.time()

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
        
    X_agglo_reduced = X[:, selected_features]
    print(f"Feature Agglomeration dataset shape: {X_agglo_reduced.shape}")
    print(f"Feature Agglomeration completed in {time.time() - start_time:.2f} seconds")
    
    # Display top 5 features from Feature Agglomeration
    print("\nTop 5 Feature Agglomeration selected features:")
    for i in range(min(5, len(selected_features))):
        feature_idx = selected_features[i]
        cluster_id = feature_labels[feature_idx]
        pixel_row = feature_idx // 28
        pixel_col = feature_idx % 28
        print(f"Feature {feature_idx}: Cluster {cluster_id}, Position = ({pixel_row}, {pixel_col})")
    
except Exception as e:
    print(f"Error running Feature Agglomeration: {e}")
    print("Continuing without Feature Agglomeration feature selection.")
    agglo_available = False
    X_agglo_reduced = None

# Cross-validation setup
print("\nPerforming cross-validation on reduced datasets...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. LASSO feature selection + LASSO classifier (using LogisticRegression with L1 penalty)
print(f"\nCross-validating LASSO-selected features with LASSO classifier...")
start_time = time.time()
lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
lasso_scores = cross_val_score(lasso_clf, X_lasso_reduced, y, cv=cv, scoring='accuracy')
print(f"LASSO top {num_features} features (with LASSO classifier) - CV Accuracy: {lasso_scores.mean():.4f} ± {lasso_scores.std():.4f}")
print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds")

# 2. Logistic Regression feature selection + Logistic Regression
print(f"\nCross-validating Logistic Regression-selected features with Logistic Regression...")
start_time = time.time()
logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)
logreg_scores = cross_val_score(logreg_classifier, X_logreg_reduced, y, cv=cv, scoring='accuracy')
print(f"LogReg top {num_features} features (with Logistic Regression) - CV Accuracy: {logreg_scores.mean():.4f} ± {logreg_scores.std():.4f}")
print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds")

# 3. PCA feature selection + Random Forest
print(f"\nCross-validating PCA-selected features with Random Forest...")
start_time = time.time()
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
pca_rf_scores = cross_val_score(rf_classifier, X_pca_reduced, y, cv=cv, scoring='accuracy')
print(f"PCA top {num_features} features (with Random Forest) - CV Accuracy: {pca_rf_scores.mean():.4f} ± {pca_rf_scores.std():.4f}")
print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds")

# 4. Feature Agglomeration feature selection + Random Forest
if agglo_available and X_agglo_reduced is not None:
    print(f"\nCross-validating Feature Agglomeration-selected features with Random Forest...")
    start_time = time.time()
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    agglo_rf_scores = cross_val_score(rf_classifier, X_agglo_reduced, y, cv=cv, scoring='accuracy')
    print(f"Feature Agglomeration top {num_features} features (with Random Forest) - CV Accuracy: {agglo_rf_scores.mean():.4f} ± {agglo_rf_scores.std():.4f}")
    print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds")
