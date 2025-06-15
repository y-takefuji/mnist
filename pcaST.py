import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load MNIST dataset (70,000 samples, 784 features)
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float64')
y = mnist.target.astype('int')

print(f"Dataset shape: {X.shape}")  # Should be (70000, 784)

# Create a pipeline with scaling and PCA to extract top 30 features
print("Applying PCA with scaling to extract top 30 features...")
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30))
])

# Fit PCA and transform the data to get the reduced dataset with top 30 features
X_reduced = pca_pipeline.fit_transform(X)
print(f"Reduced data shape: {X_reduced.shape}")  # Should be (70000, 30)

# Get the PCA model from the pipeline
pca = pca_pipeline.named_steps['pca']

# Print explained variance information
explained_variance = pca.explained_variance_ratio_
print(f"Total explained variance with 30 components: {np.sum(explained_variance)*100:.2f}%")
print(f"Individual explained variance of top 5 components: {explained_variance[:5]*100}")

# Cross-validation on the reduced features (X_reduced)
print("\nPerforming cross-validation with Random Forest on reduced dataset (30 features)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(clf, X_reduced, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Print information about the top 5 principal components (combined features)
print("\nTop 5 Combined Features (Principal Components):")
for i in range(5):
    print(f"PC {i+1}: Explains {explained_variance[i]*100:.2f}% of variance")
    
    # Show the top 5 original features that contribute most to this component
    component = pca.components_[i]
    top_indices = np.argsort(np.abs(component))[-5:]  # Get indices of 5 highest absolute values
    top_values = component[top_indices]
    print(f"  Top 5 contributing original features (indices and weights):")
    for idx, val in zip(top_indices, top_values):
        print(f"    Feature {idx}: {val:.4f}")
    print()

# Print cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)
print("\nCumulative Variance for Top 5 Components:")
for i in range(5):
    print(f"PC 1-{i+1}: {cumulative_variance[i]*100:.2f}% of total variance")
