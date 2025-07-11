"""
K-Means Clustering Implementation from Scratch

This module implements K-Means clustering algorithm with various initialization
methods and distance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal, Tuple
import warnings


class KMeans:
    """
    K-Means clustering implementation from scratch.
    
    Parameters:
    -----------
    n_clusters : int, default=8
        Number of clusters to form
    init : str, default='k-means++'
        Initialization method: 'k-means++', 'random', or 'manual'
    max_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++',
                 max_iter: int = 300, tol: float = 1e-4,
                 random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Fitted attributes
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.n_features_ = None
        
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centroids.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data
            
        Returns:
        --------
        centroids : np.ndarray
            Initial centroids
        """
        n_samples, n_features = X.shape
        
        if self.random_state:
            np.random.seed(self.random_state)
        
        if self.init == 'random':
            # Random initialization
            centroids = np.random.randn(self.n_clusters, n_features)
            
            # Scale to data range
            for i in range(n_features):
                min_val, max_val = X[:, i].min(), X[:, i].max()
                centroids[:, i] = centroids[:, i] * (max_val - min_val) + min_val
                
        elif self.init == 'k-means++':
            # K-means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            centroids[0] = X[np.random.choice(n_samples)]
            
            # Choose remaining centroids
            for i in range(1, self.n_clusters):
                # Calculate distances to nearest centroid
                distances = np.inf * np.ones(n_samples)
                
                for j in range(n_samples):
                    for k in range(i):
                        dist = np.linalg.norm(X[j] - centroids[k])
                        distances[j] = min(distances[j], dist)
                
                # Choose next centroid with probability proportional to squared distance
                probabilities = distances ** 2
                probabilities /= probabilities.sum()
                
                cumulative_probs = probabilities.cumsum()
                r = np.random.random()
                
                for j in range(n_samples):
                    if r < cumulative_probs[j]:
                        centroids[i] = X[j]
                        break
                        
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to the closest centroid.
        
        Parameters:
        -----------
        X : np.ndarray
            Data points
        centroids : np.ndarray
            Current centroids
            
        Returns:
        --------
        labels : np.ndarray
            Cluster assignments
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids based on current cluster assignments.
        
        Parameters:
        -----------
        X : np.ndarray
            Data points
        labels : np.ndarray
            Current cluster assignments
            
        Returns:
        --------
        centroids : np.ndarray
            Updated centroids
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[i] = X[np.random.choice(len(X))]
                warnings.warn(f"Cluster {i} is empty. Reinitializing centroid.")
        
        return centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (inertia).
        
        Parameters:
        -----------
        X : np.ndarray
            Data points
        labels : np.ndarray
            Cluster assignments
        centroids : np.ndarray
            Centroids
            
        Returns:
        --------
        inertia : float
            Within-cluster sum of squares
        """
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)
        
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-Means clustering to the data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : KMeans
            Returns self for method chaining
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        if self.n_clusters > n_samples:
            raise ValueError(f"Number of clusters ({self.n_clusters}) cannot be greater than number of samples ({n_samples})")
        
        # Initialize centroids
        centroids = self._init_centroids(X)
        
        # Main K-means loop
        for iteration in range(self.max_iter):
            # Assign points to clusters
            labels = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            
            if centroid_shift < self.tol:
                break
            
            centroids = new_centroids
        
        # Store results
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = self._calculate_inertia(X, labels, centroids)
        self.n_iter_ = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        labels : np.ndarray
            Predicted cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict cluster labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data
            
        Returns:
        --------
        labels : np.ndarray
            Cluster labels
        """
        self.fit(X)
        return self.labels_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to cluster-distance space.
        
        Parameters:
        -----------
        X : np.ndarray
            Data to transform
            
        Returns:
        --------
        distances : np.ndarray
            Distances to each cluster center
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                distances[i, j] = np.linalg.norm(X[i] - self.cluster_centers_[j])
        
        return distances
    
    def score(self, X: np.ndarray) -> float:
        """
        Return the negative inertia (higher is better).
        
        Parameters:
        -----------
        X : np.ndarray
            Data to score
            
        Returns:
        --------
        score : float
            Negative inertia
        """
        labels = self.predict(X)
        inertia = self._calculate_inertia(X, labels, self.cluster_centers_)
        return -inertia


def elbow_method(X: np.ndarray, max_k: int = 10, random_state: Optional[int] = None) -> Tuple[list, list]:
    """
    Perform elbow method to find optimal number of clusters.
    
    Parameters:
    -----------
    X : np.ndarray
        Data to cluster
    max_k : int
        Maximum number of clusters to try
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    k_values, inertias : tuple
        K values and corresponding inertias
    """
    k_values = range(1, max_k + 1)
    inertias = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return list(k_values), inertias


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate silhouette score for clustering evaluation.
    
    Parameters:
    -----------
    X : np.ndarray
        Data points
    labels : np.ndarray
        Cluster labels
        
    Returns:
    --------
    silhouette_score : float
        Average silhouette score
    """
    n_samples = len(X)
    n_clusters = len(np.unique(labels))
    
    if n_clusters == 1:
        return 0.0
    
    silhouette_scores = []
    
    for i in range(n_samples):
        # Calculate a(i): average distance to points in same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) == 1:
            a_i = 0.0
        else:
            a_i = np.mean([np.linalg.norm(X[i] - point) for point in same_cluster if not np.array_equal(point, X[i])])
        
        # Calculate b(i): minimum average distance to points in other clusters
        b_i = float('inf')
        for cluster_label in np.unique(labels):
            if cluster_label != labels[i]:
                other_cluster = X[labels == cluster_label]
                if len(other_cluster) > 0:
                    avg_dist = np.mean([np.linalg.norm(X[i] - point) for point in other_cluster])
                    b_i = min(b_i, avg_dist)
        
        # Calculate silhouette score for point i
        if max(a_i, b_i) == 0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)


def generate_cluster_data(n_samples: int = 300, n_centers: int = 4,
                         n_features: int = 2, cluster_std: float = 1.0,
                         random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic clustering data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_centers : int
        Number of cluster centers
    n_features : int
        Number of features
    cluster_std : float
        Standard deviation of clusters
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    X, y : tuple
        Features and true cluster labels
    """
    if random_state:
        np.random.seed(random_state)
    
    # Generate cluster centers
    centers = np.random.uniform(-5, 5, (n_centers, n_features))
    
    # Generate samples for each cluster
    samples_per_cluster = n_samples // n_centers
    X = []
    y = []
    
    for i in range(n_centers):
        cluster_samples = np.random.multivariate_normal(
            centers[i], np.eye(n_features) * cluster_std**2, samples_per_cluster
        )
        X.append(cluster_samples)
        y.extend([i] * samples_per_cluster)
    
    # Add remaining samples to last cluster
    remaining = n_samples - samples_per_cluster * n_centers
    if remaining > 0:
        cluster_samples = np.random.multivariate_normal(
            centers[-1], np.eye(n_features) * cluster_std**2, remaining
        )
        X.append(cluster_samples)
        y.extend([-1] * remaining)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


class KMeansVisualization:
    """
    Visualization utilities for K-Means clustering.
    """
    
    @staticmethod
    def plot_clusters(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                     title: str = "K-Means Clustering Results") -> None:
        """
        Plot clustering results for 2D data.
        
        Parameters:
        -----------
        X : np.ndarray
            Data points
        labels : np.ndarray
            Cluster labels
        centroids : np.ndarray
            Cluster centroids
        title : str
            Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Can only plot 2D data")
        
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            cluster_points = X[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
        
        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='red', marker='x', s=200, linewidth=3, label='Centroids')
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_elbow_curve(k_values: list, inertias: list) -> None:
        """
        Plot elbow curve for determining optimal number of clusters.
        
        Parameters:
        -----------
        k_values : list
            Number of clusters
        inertias : list
            Corresponding inertias
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, 'bo-', markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Within-cluster sum of squares)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True, alpha=0.3)
        
        # Highlight potential elbow points
        if len(inertias) > 2:
            # Calculate second derivative to find elbow
            second_deriv = np.diff(inertias, n=2)
            if len(second_deriv) > 0:
                elbow_idx = np.argmax(second_deriv) + 2
                if elbow_idx < len(k_values):
                    plt.axvline(x=k_values[elbow_idx], color='red', linestyle='--', 
                               alpha=0.7, label=f'Potential elbow at k={k_values[elbow_idx]}')
                    plt.legend()
        
        plt.show()
    
    @staticmethod
    def plot_convergence(kmeans: KMeans, X: np.ndarray, max_iter: int = 20) -> None:
        """
        Visualize K-Means convergence process.
        
        Parameters:
        -----------
        kmeans : KMeans
            K-Means instance (will be re-fitted)
        X : np.ndarray
            Data to cluster
        max_iter : int
            Maximum iterations to show
        """
        if X.shape[1] != 2:
            raise ValueError("Can only visualize 2D data")
        
        # Re-fit with limited iterations to show convergence
        kmeans_viz = KMeans(n_clusters=kmeans.n_clusters, init=kmeans.init,
                           max_iter=1, random_state=kmeans.random_state)
        
        # Store convergence history
        centroid_history = []
        inertia_history = []
        
        X_copy = X.copy()
        
        for i in range(max_iter):
            kmeans_viz.fit(X_copy)
            centroid_history.append(kmeans_viz.cluster_centers_.copy())
            inertia_history.append(kmeans_viz.inertia_)
            
            # Update for next iteration
            kmeans_viz.max_iter = 1
            if i > 0:
                kmeans_viz.cluster_centers_ = centroid_history[-1]
        
        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot final clustering result
        final_labels = kmeans_viz.predict(X)
        unique_labels = np.unique(final_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            cluster_points = X[final_labels == label]
            ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
        
        # Plot centroid evolution
        for i in range(len(centroid_history)):
            centroids = centroid_history[i]
            alpha = 0.3 + 0.7 * i / len(centroid_history)
            ax1.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='x', s=100, alpha=alpha)
        
        ax1.set_title('Centroid Evolution')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot inertia evolution
        ax2.plot(range(len(inertia_history)), inertia_history, 'bo-')
        ax2.set_title('Inertia Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Inertia')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
