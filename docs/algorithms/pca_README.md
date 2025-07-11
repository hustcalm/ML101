# Principal Component Analysis (PCA)

Principal Component Analysis is a dimensionality reduction technique that finds the directions of maximum variance in high-dimensional data and projects the data onto these directions to reduce dimensionality while preserving as much information as possible.

## Mathematical Foundation

### The PCA Problem

Given a dataset X with n samples and p features, PCA finds a new coordinate system where:
1. The first coordinate (PC1) captures the maximum variance
2. The second coordinate (PC2) captures the maximum remaining variance
3. And so on...

### Mathematical Formulation

**Step 1: Center the Data**
```
X_centered = X - mean(X)
```

**Step 2: Compute Covariance Matrix**
```
C = (1/(n-1)) * X_centered^T * X_centered
```

**Step 3: Eigenvalue Decomposition**
```
C = V * Λ * V^T
```
Where V contains eigenvectors (principal components) and Λ contains eigenvalues.

**Step 4: Project Data**
```
X_reduced = X_centered * V[:, :k]
```
Where k is the number of components to keep.

### Key Concepts

1. **Principal Components**: Eigenvectors of the covariance matrix
2. **Explained Variance**: Eigenvalues represent variance captured by each component
3. **Dimensionality Reduction**: Keep only the top k components
4. **Reconstruction**: Project back to original space

## Implementation Details

### Algorithm Steps

1. **Standardize/Center** the data
2. **Compute covariance matrix** C = X^T X / (n-1)
3. **Find eigenvalues and eigenvectors** of C
4. **Sort by eigenvalues** (descending order)
5. **Select top k components** based on explained variance
6. **Transform data** by projecting onto selected components

### Choosing Number of Components

Three common approaches:
1. **Fixed number**: Specify exact number of components
2. **Variance threshold**: Keep components explaining X% of variance
3. **Elbow method**: Look for "elbow" in explained variance plot

## Usage Example

```python
from pca import PCA, generate_sample_data
import numpy as np

# Generate sample data
X, y = generate_sample_data(n_samples=200, n_features=20, random_state=42)

# Method 1: Fixed number of components
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_}")

# Method 2: Variance threshold
pca_var = PCA(n_components=0.95)  # Keep 95% of variance
X_reduced_var = pca_var.fit_transform(X)
print(f"Components for 95% variance: {X_reduced_var.shape[1]}")

# Reconstruction
X_reconstructed = pca.inverse_transform(X_reduced)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"Reconstruction error: {reconstruction_error:.6f}")

# Visualizations
pca.plot_explained_variance()
pca.plot_components(n_components=3)
pca.plot_2d_projection(X, y)
```

## Parameters

- **n_components**: Number of components to keep (int, float, or None)
- **whiten**: Whether to scale components by sqrt(eigenvalues)
- **random_state**: Random seed for reproducibility

## Attributes

- **components_**: Principal components (eigenvectors)
- **explained_variance_**: Variance explained by each component
- **explained_variance_ratio_**: Proportion of variance explained
- **mean_**: Mean of training data
- **singular_values_**: Singular values for compatibility

## Advantages

1. **Dimensionality Reduction**: Reduces computational cost
2. **Noise Reduction**: Removes low-variance directions
3. **Visualization**: Projects to 2D/3D for plotting
4. **Feature Extraction**: Creates uncorrelated features
5. **Data Compression**: Reduces storage requirements

## Disadvantages

1. **Interpretability**: Components are linear combinations of features
2. **Linear Assumption**: Only captures linear relationships
3. **Standardization Sensitive**: Requires feature scaling
4. **Information Loss**: Discards some variance
5. **Computational Cost**: O(p³) for eigenvalue decomposition

## Key Concepts Explained

### Explained Variance Ratio
Shows how much variance each component captures:
```python
# First component might capture 40% of variance
# Second component might capture 25% of variance
# Third component might capture 15% of variance
# etc.
```

### Cumulative Variance
Running total of explained variance:
```python
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# Often aim for 80-95% total variance
```

### Reconstruction Error
Measures information loss from dimensionality reduction:
```python
error = np.mean((X_original - X_reconstructed) ** 2)
# Lower error = better reconstruction
```

## Applications

### 1. Dimensionality Reduction
```python
# Reduce 1000 features to 50 features
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_high_dim)
```

### 2. Data Visualization
```python
# Project to 2D for plotting
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
```

### 3. Noise Reduction
```python
# Keep only components with high variance
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_denoised = pca.fit_transform(X)
```

### 4. Feature Extraction
```python
# Create uncorrelated features
pca = PCA(n_components=10)
features = pca.fit_transform(X)
# Use these features for machine learning
```

## Practical Tips

### 1. Standardize Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA().fit(X_scaled)
```

### 2. Choose Number of Components
```python
# Plot explained variance to decide
pca = PCA()
pca.fit(X)
pca.plot_explained_variance()

# Common thresholds: 80%, 90%, 95%
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumvar >= 0.95) + 1
```

### 3. Interpret Components
```python
# Examine component loadings
pca = PCA(n_components=3)
pca.fit(X)
pca.plot_components(feature_names=feature_names)
```

## Comparison with Other Methods

| Method | Type | Pros | Cons |
|--------|------|------|------|
| PCA | Linear | Fast, interpretable | Linear only |
| t-SNE | Non-linear | Good for visualization | Slow, not for ML |
| UMAP | Non-linear | Fast, preserves structure | Less interpretable |
| Autoencoders | Non-linear | Very flexible | Requires training |

## Mathematical Intuition

Think of PCA as finding the "best" camera angles to photograph a 3D object:
- The first angle (PC1) shows the most variation
- The second angle (PC2) shows the most remaining variation
- Each angle is perpendicular to the others

For data, PCA finds the directions where points are most spread out, which contain the most information about the data structure.

## Common Pitfalls

1. **Forgetting to standardize**: Features with different scales dominate
2. **Over-reducing**: Losing too much information
3. **Interpreting components**: Linear combinations can be hard to understand
4. **Assuming linearity**: PCA only finds linear relationships

## Validation

```python
# Cross-validation for optimal number of components
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = []
for n in range(1, 21):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    
    clf = LogisticRegression()
    score = cross_val_score(clf, X_pca, y, cv=5).mean()
    scores.append(score)

optimal_n = np.argmax(scores) + 1
```
