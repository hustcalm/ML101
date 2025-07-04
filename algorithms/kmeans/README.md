# K-Means Clustering

K-Means is one of the most popular unsupervised machine learning algorithms for clustering. It partitions data into k clusters by minimizing the within-cluster sum of squares (WCSS), also known as inertia.

## Mathematical Foundation

### Objective Function

K-Means aims to minimize the within-cluster sum of squares:

```
J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

Where:
- k: Number of clusters
- Cᵢ: Set of points in cluster i
- μᵢ: Centroid of cluster i
- ||x - μᵢ||²: Squared Euclidean distance between point x and centroid μᵢ

### Algorithm Steps

1. **Initialize**: Choose k initial centroids (randomly or using K-means++)
2. **Assignment**: Assign each point to the nearest centroid
3. **Update**: Recalculate centroids as the mean of assigned points
4. **Repeat**: Continue steps 2-3 until convergence

### Centroid Update

The centroid of each cluster is updated as:

```
μᵢ = (1/|Cᵢ|) * Σₓ∈Cᵢ x
```

Where |Cᵢ| is the number of points in cluster i.

## Key Properties

### Initialization Methods

**Random Initialization:**
- Randomly place centroids within the data range
- Simple but can lead to poor local optima

**K-Means++ Initialization:**
- Choose first centroid randomly
- Choose subsequent centroids with probability proportional to squared distance from nearest existing centroid
- Leads to better initial placement and faster convergence

### Convergence

The algorithm converges when:
- Centroids move less than a tolerance threshold
- Maximum number of iterations is reached
- Cluster assignments no longer change

## Advantages

1. **Simple and Fast**: Easy to understand and implement
2. **Scalable**: Works well with large datasets
3. **Guaranteed Convergence**: Always converges to a local optimum
4. **Memory Efficient**: Low memory requirements
5. **Works Well with Globular Clusters**: Effective for spherical, well-separated clusters

## Disadvantages

1. **Requires Choosing k**: Must specify number of clusters beforehand
2. **Sensitive to Initialization**: Can converge to poor local optima
3. **Assumes Spherical Clusters**: Struggles with non-spherical or varying-size clusters
4. **Sensitive to Outliers**: Outliers can significantly affect centroids
5. **Sensitive to Feature Scaling**: Features with larger scales dominate the distance calculation

## Implementation Features

Our implementation includes:

- **Multiple Initialization Methods**: Random and K-means++
- **Convergence Monitoring**: Tracks centroid movement and iterations
- **Inertia Calculation**: Provides clustering quality metric
- **Flexible Interface**: Supports fit, predict, and transform methods
- **Robust Handling**: Manages empty clusters and edge cases

## Usage Example

```python
from algorithms.kmeans.kmeans import KMeans
import numpy as np

# Generate sample data
X = np.random.randn(100, 2)

# Create and fit K-Means model
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_
print(f"Cluster labels: {labels}")

# Get cluster centers
centers = kmeans.cluster_centers_
print(f"Cluster centers: {centers}")

# Predict new data
new_data = np.random.randn(10, 2)
new_labels = kmeans.predict(new_data)
print(f"New predictions: {new_labels}")

# Get distances to cluster centers
distances = kmeans.transform(new_data)
print(f"Distances to centers: {distances}")
```

## Parameters

- **n_clusters** (int, default=8): Number of clusters to form
- **init** (str, default='k-means++'): Initialization method ('k-means++' or 'random')
- **max_iter** (int, default=300): Maximum number of iterations
- **tol** (float, default=1e-4): Tolerance for convergence
- **random_state** (int, default=None): Random seed for reproducibility

## Attributes

- **cluster_centers_**: Coordinates of cluster centers
- **labels_**: Cluster labels for each point in the training data
- **inertia_**: Sum of squared distances of samples to their closest cluster center
- **n_iter_**: Number of iterations run
- **n_features_**: Number of features in the training data

## Choosing the Number of Clusters

### Elbow Method

Plot inertia vs. number of clusters and look for the "elbow" point:

```python
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
```

### Silhouette Analysis

Use silhouette score to evaluate cluster quality:

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Choose k with highest silhouette score
```

## Applications

1. **Customer Segmentation**: Group customers by behavior or demographics
2. **Image Segmentation**: Segment images into regions
3. **Market Research**: Identify distinct market segments
4. **Data Compression**: Reduce data by representing clusters with centroids
5. **Anomaly Detection**: Identify outliers based on distance to cluster centers
6. **Gene Sequencing**: Cluster genes with similar expression patterns

## Comparison with Other Clustering Methods

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| K-Means | Fast, simple, scalable | Requires k, assumes spherical clusters |
| Hierarchical | No need to specify k, creates dendrograms | Computationally expensive O(n³) |
| DBSCAN | Finds arbitrary shapes, handles outliers | Sensitive to parameters, struggles with varying densities |
| GMM | Soft clustering, handles overlapping clusters | More complex, requires choosing components |

## Tips for Better Results

1. **Feature Scaling**: Normalize features to have similar scales
2. **Dimensionality Reduction**: Use PCA for high-dimensional data
3. **Multiple Runs**: Run algorithm multiple times with different initializations
4. **Domain Knowledge**: Use domain expertise to choose appropriate k
5. **Data Preprocessing**: Remove outliers and handle missing values

## Mathematical Complexity

- **Time Complexity**: O(n × k × i × d)
  - n: number of samples
  - k: number of clusters  
  - i: number of iterations
  - d: number of dimensions

- **Space Complexity**: O(n × d + k × d)
  - Storage for data and centroids

## References

1. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations
2. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding
3. Lloyd, S. (1982). Least squares quantization in PCM