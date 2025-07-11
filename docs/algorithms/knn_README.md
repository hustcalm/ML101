# K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple, versatile, and intuitive machine learning algorithm that can be used for both classification and regression tasks. It's a non-parametric, lazy learning algorithm that makes predictions based on the k nearest neighbors in the feature space.

## Mathematical Foundation

### Core Principle

KNN is based on the assumption that similar data points are likely to have similar labels or values. The algorithm finds the k nearest neighbors to a query point and makes predictions based on these neighbors.

### Distance Metrics

**Euclidean Distance (L2 norm):**
```
d(x, y) = √(Σᵢ₌₁ⁿ (xᵢ - yᵢ)²)
```

**Manhattan Distance (L1 norm):**
```
d(x, y) = Σᵢ₌₁ⁿ |xᵢ - yᵢ|
```

**Minkowski Distance:**
```
d(x, y) = (Σᵢ₌₁ⁿ |xᵢ - yᵢ|ᵖ)^(1/p)
```

Where p=1 gives Manhattan distance and p=2 gives Euclidean distance.

### Prediction Methods

**Classification:**
- **Majority Voting**: Assign the most common class among k neighbors
- **Weighted Voting**: Weight votes by inverse distance to give closer neighbors more influence

**Regression:**
- **Mean**: Average of k nearest neighbor values
- **Weighted Mean**: Distance-weighted average of neighbor values

## Key Properties

### Lazy Learning

KNN is called a "lazy" algorithm because:
- No explicit training phase
- All computation happens during prediction
- Simply stores the training data

### Non-parametric

KNN makes no assumptions about the underlying data distribution, making it flexible for various data types and patterns.

### Instance-based Learning

Predictions are made based on specific instances (training examples) rather than learned parameters.

## Advantages

1. **Simple and Intuitive**: Easy to understand and implement
2. **No Training Required**: No explicit model training phase
3. **Versatile**: Works for both classification and regression
4. **Non-parametric**: No assumptions about data distribution
5. **Naturally Handles Multi-class**: Can handle multiple classes without modification
6. **Local Patterns**: Adapts to local patterns in the data
7. **Online Learning**: Can easily incorporate new data

## Disadvantages

1. **Computationally Expensive**: O(n) time complexity for each prediction
2. **Sensitive to Irrelevant Features**: All features contribute to distance calculation
3. **Curse of Dimensionality**: Performance degrades in high-dimensional spaces
4. **Sensitive to Scale**: Features with larger scales dominate distance calculations
5. **Memory Intensive**: Must store all training data
6. **Sensitive to Local Structure**: Can be influenced by noisy or outlier data
7. **Requires Choosing k**: Performance depends on the choice of k

## Implementation Features

Our implementation includes:

- **Multiple Distance Metrics**: Euclidean, Manhattan, and Minkowski distances
- **Flexible Tasks**: Both classification and regression support
- **Weighting Schemes**: Uniform and distance-based weighting
- **Probability Estimates**: Class probability predictions for classification
- **Robust Implementation**: Handles edge cases and provides clear error messages

## Usage Example

### Classification

```python
from algorithms.knn.knn import KNearestNeighbors
import numpy as np

# Generate sample data
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

# Create and fit KNN classifier
knn = KNearestNeighbors(k=5, task='classification', distance_metric='euclidean')
knn.fit(X_train, y_train)

# Make predictions
X_test = np.random.randn(20, 2)
predictions = knn.predict(X_test)
print(f"Predictions: {predictions}")

# Get class probabilities
probabilities = knn.predict_proba(X_test)
print(f"Class probabilities: {probabilities}")
```

### Regression

```python
# Generate regression data
X_train = np.random.randn(100, 2)
y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1

# Create and fit KNN regressor
knn = KNearestNeighbors(k=5, task='regression', weights='distance')
knn.fit(X_train, y_train)

# Make predictions
X_test = np.random.randn(20, 2)
predictions = knn.predict(X_test)
print(f"Regression predictions: {predictions}")
```

## Parameters

- **k** (int, default=5): Number of neighbors to consider
- **task** (str, default='classification'): Task type ('classification' or 'regression')
- **distance_metric** (str, default='euclidean'): Distance metric ('euclidean', 'manhattan', 'minkowski')
- **p** (float, default=2): Parameter for Minkowski distance
- **weights** (str, default='uniform'): Weighting scheme ('uniform' or 'distance')

## Attributes

- **classes_**: Unique class labels (for classification tasks)
- **X_train**: Stored training features
- **y_train**: Stored training labels/values

## Choosing the Optimal k

### Odd vs Even k

For binary classification, use odd k to avoid ties:
- k=3, 5, 7, 9, etc.

### Cross-Validation

Use cross-validation to find the optimal k:

```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNearestNeighbors(k=k, task='classification')
    knn.fit(X_train, y_train)
    # Implement cross-validation scoring
    score = evaluate_model(knn, X_train, y_train)
    cv_scores.append(score)

# Plot k vs accuracy
plt.plot(k_values, cv_scores, 'bo-')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Accuracy vs k')
plt.show()

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")
```

### Rule of Thumb

A common heuristic is to use k = √n, where n is the number of training samples.

## Feature Scaling

KNN is sensitive to feature scaling. Always normalize features:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNearestNeighbors(k=5)
knn.fit(X_train_scaled, y_train)
predictions = knn.predict(X_test_scaled)
```

## Applications

1. **Recommendation Systems**: Find similar users or items
2. **Pattern Recognition**: Handwriting recognition, image classification
3. **Anomaly Detection**: Identify outliers based on distance to neighbors
4. **Gene Classification**: Classify genes based on expression patterns
5. **Text Mining**: Document classification and clustering
6. **Market Research**: Customer segmentation and behavior analysis
7. **Medical Diagnosis**: Classify medical conditions based on symptoms

## Comparison with Other Algorithms

| Algorithm | Advantages | Disadvantages |
|-----------|------------|---------------|
| KNN | Simple, no training, versatile | Computationally expensive, sensitive to irrelevant features |
| Decision Trees | Fast prediction, interpretable | Can overfit, biased toward features with many levels |
| SVM | Effective in high dimensions, memory efficient | Requires feature scaling, no probability estimates |
| Naive Bayes | Fast, works well with small datasets | Strong independence assumption |
| Neural Networks | Can learn complex patterns | Requires large datasets, black box |

## Optimization Techniques

### Dimensionality Reduction

Use PCA or other techniques to reduce dimensionality:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

### Feature Selection

Remove irrelevant features to improve performance:

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

### Approximate Nearest Neighbors

For large datasets, consider approximate methods:
- Locality-Sensitive Hashing (LSH)
- k-d trees
- Ball trees

## Mathematical Complexity

- **Time Complexity**: 
  - Training: O(1) (lazy learning)
  - Prediction: O(n × d) per query
  - n: number of training samples
  - d: number of features

- **Space Complexity**: O(n × d)
  - Must store all training data

## Variants

1. **k-d Trees**: Efficient for low-dimensional data
2. **Ball Trees**: Better for high-dimensional data
3. **LSH**: Approximate nearest neighbors for very large datasets
4. **Weighted KNN**: Distance-based weighting of neighbors
5. **Adaptive KNN**: Dynamically adjust k based on local density

## Tips for Better Results

1. **Normalize Features**: Use StandardScaler or MinMaxScaler
2. **Remove Irrelevant Features**: Use feature selection techniques
3. **Handle Missing Values**: Impute or remove missing data
4. **Choose Appropriate k**: Use cross-validation
5. **Consider Distance Metric**: Manhattan for high-dimensional data
6. **Handle Imbalanced Data**: Use weighted voting or sampling techniques

## References

1. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification
2. Fix, E., & Hodges, J. L. (1951). Discriminatory analysis: nonparametric discrimination
3. Altman, N. S. (1992). An introduction to kernel and nearest-neighbor nonparametric regression