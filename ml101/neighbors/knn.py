"""
K-Nearest Neighbors (KNN) Implementation from Scratch

This module implements KNN for both classification and regression tasks
with different distance metrics and weighting schemes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Literal
from collections import Counter


class KNearestNeighbors:
    """
    K-Nearest Neighbors implementation from scratch.
    
    Supports both classification and regression with various distance metrics.
    
    Parameters:
    -----------
    k : int, default=5
        Number of neighbors to consider
    task : str, default='classification'
        Task type: 'classification' or 'regression'
    distance_metric : str, default='euclidean'
        Distance metric: 'euclidean', 'manhattan', 'minkowski'
    p : float, default=2
        Parameter for Minkowski distance (p=1: Manhattan, p=2: Euclidean)
    weights : str, default='uniform'
        Weighting scheme: 'uniform' or 'distance'
    """
    
    def __init__(self, k: int = 5, 
                 task: Literal['classification', 'regression'] = 'classification',
                 distance_metric: str = 'euclidean',
                 p: float = 2,
                 weights: str = 'uniform'):
        self.k = k
        self.task = task
        self.distance_metric = distance_metric
        self.p = p
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestNeighbors':
        """
        Fit the KNN model (store training data).
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : KNearestNeighbors
            Returns self for method chaining
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            
        return self
    
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute distance between two points.
        
        Parameters:
        -----------
        x1, x2 : np.ndarray
            Points to compute distance between
            
        Returns:
        --------
        distance : float
            Distance between points
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """
        Find k nearest neighbors for a single point.
        
        Parameters:
        -----------
        x : np.ndarray
            Query point
            
        Returns:
        --------
        neighbor_indices, distances : tuple
            Indices of neighbors and their distances
        """
        if self.X_train is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        # Compute distances to all training points
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._compute_distance(x, x_train)
            distances.append((dist, i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        neighbor_distances = np.array([d[0] for d in k_nearest])
        neighbor_indices = np.array([d[1] for d in k_nearest])
        
        return neighbor_indices, neighbor_distances
    
    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute weights for neighbors based on distances.
        
        Parameters:
        -----------
        distances : np.ndarray
            Distances to neighbors
            
        Returns:
        --------
        weights : np.ndarray
            Weights for each neighbor
        """
        if self.weights == 'uniform':
            return np.ones(len(distances))
        elif self.weights == 'distance':
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            return 1 / distances
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values or class labels
        """
        if self.X_train is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        predictions = []
        
        for x in X:
            neighbor_indices, distances = self._get_neighbors(x)
            neighbor_labels = self.y_train[neighbor_indices]
            weights = self._compute_weights(distances)
            
            if self.task == 'classification':
                prediction = self._predict_classification(neighbor_labels, weights)
            else:  # regression
                prediction = self._predict_regression(neighbor_labels, weights)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _predict_classification(self, neighbor_labels: np.ndarray, 
                              weights: np.ndarray) -> Union[int, str]:
        """
        Predict class label using weighted voting.
        
        Parameters:
        -----------
        neighbor_labels : np.ndarray
            Labels of k nearest neighbors
        weights : np.ndarray
            Weights for each neighbor
            
        Returns:
        --------
        prediction : Union[int, str]
            Predicted class label
        """
        if self.weights == 'uniform':
            # Simple majority voting
            vote_counts = Counter(neighbor_labels)
            return vote_counts.most_common(1)[0][0]
        else:
            # Weighted voting
            class_weights = {}
            for label, weight in zip(neighbor_labels, weights):
                if label in class_weights:
                    class_weights[label] += weight
                else:
                    class_weights[label] = weight
            
            return max(class_weights, key=class_weights.get)
    
    def _predict_regression(self, neighbor_values: np.ndarray, 
                          weights: np.ndarray) -> float:
        """
        Predict continuous value using weighted average.
        
        Parameters:
        -----------
        neighbor_values : np.ndarray
            Values of k nearest neighbors
        weights : np.ndarray
            Weights for each neighbor
            
        Returns:
        --------
        prediction : float
            Predicted continuous value
        """
        return np.average(neighbor_values, weights=weights)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if self.X_train is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        n_classes = len(self.classes_)
        probabilities = []
        
        for x in X:
            neighbor_indices, distances = self._get_neighbors(x)
            neighbor_labels = self.y_train[neighbor_indices]
            weights = self._compute_weights(distances)
            
            # Compute class probabilities
            class_probs = np.zeros(n_classes)
            total_weight = np.sum(weights)
            
            for label, weight in zip(neighbor_labels, weights):
                class_idx = np.where(self.classes_ == label)[0][0]
                class_probs[class_idx] += weight
            
            class_probs /= total_weight
            probabilities.append(class_probs)
        
        return np.array(probabilities)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy (classification) or R² score (regression).
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True target values
            
        Returns:
        --------
        score : float
            Accuracy or R² score
        """
        predictions = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(predictions == y)
        else:  # regression
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)


def generate_knn_data(task: str = 'classification', n_samples: int = 200, 
                     noise: float = 0.1, random_state: Optional[int] = None) -> tuple:
    """
    Generate synthetic data for KNN examples.
    
    Parameters:
    -----------
    task : str
        'classification' or 'regression'
    n_samples : int
        Number of samples
    noise : float
        Noise level
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    X, y : tuple
        Features and targets
    """
    if random_state:
        np.random.seed(random_state)
    
    if task == 'classification':
        # Generate 2D classification data with 3 classes
        centers = np.array([[0, 0], [3, 3], [-2, 2]])
        X = []
        y = []
        
        samples_per_class = n_samples // 3
        for i, center in enumerate(centers):
            class_samples = np.random.randn(samples_per_class, 2) * noise + center
            X.append(class_samples)
            y.extend([i] * samples_per_class)
        
        X = np.vstack(X)
        y = np.array(y)
        
        # Shuffle
        indices = np.random.permutation(len(y))
        return X[indices], y[indices]
    
    else:  # regression
        # Generate 1D regression data
        X = np.random.uniform(-3, 3, (n_samples, 1))
        y = np.sin(X.ravel()) + np.random.normal(0, noise, n_samples)
        return X, y


class KNNVisualization:
    """
    Visualization utilities for KNN.
    """
    
    @staticmethod
    def plot_decision_boundary(model: KNearestNeighbors, X: np.ndarray, y: np.ndarray,
                             h: float = 0.1, title: str = "KNN Decision Boundary"):
        """
        Plot decision boundary for 2D classification data.
        """
        if X.shape[1] != 2:
            raise ValueError("Can only plot decision boundary for 2D data")
        
        # Create a mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.title(f"{title} (k={model.k})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    
    @staticmethod
    def plot_k_analysis(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       k_range: range = range(1, 21)):
        """
        Analyze the effect of different k values.
        """
        train_scores = []
        test_scores = []
        
        for k in k_range:
            model = KNearestNeighbors(k=k)
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, train_scores, 'o-', label='Training Score', color='blue')
        plt.plot(k_range, test_scores, 'o-', label='Test Score', color='red')
        plt.xlabel('k (Number of Neighbors)')
        plt.ylabel('Accuracy')
        plt.title('KNN Performance vs k Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Find best k
        best_k = k_range[np.argmax(test_scores)]
        print(f"Best k value: {best_k} (Test Accuracy: {max(test_scores):.4f})")
        
        return best_k
