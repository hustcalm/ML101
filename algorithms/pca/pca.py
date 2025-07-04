"""
Principal Component Analysis (PCA) Implementation

This module implements Principal Component Analysis for dimensionality reduction.
PCA finds the directions of maximum variance in the data and projects the data
onto these directions to reduce dimensionality while preserving as much
information as possible.

Mathematical Foundation:
- PCA finds eigenvectors of the covariance matrix
- Principal components are eigenvectors with largest eigenvalues
- Projection: X_reduced = X_centered @ components.T
- Reconstruction: X_reconstructed = X_reduced @ components + mean

Author: ML101 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import warnings


class PCA:
    """
    Principal Component Analysis for dimensionality reduction.
    
    This implementation finds the principal components using eigenvalue decomposition
    of the covariance matrix and provides methods for dimensionality reduction
    and data reconstruction.
    """
    
    def __init__(self, 
                 n_components: Optional[Union[int, float]] = None,
                 whiten: bool = False,
                 random_state: Optional[int] = None):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int, float, or None, default=None
            Number of components to keep:
            - If int: exact number of components
            - If float (0 < n_components < 1): select components to explain this variance
            - If None: keep all components
        whiten : bool, default=False
            Whether to whiten the components (scale by sqrt of eigenvalues)
        random_state : int, optional
            Random seed for reproducibility (not used in this implementation)
        """
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        
        # Initialize attributes
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA to the data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : PCA
            Fitted estimator
        """
        X = np.array(X, dtype=np.float64)
        
        self.n_samples_, self.n_features_ = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        # Use (X.T @ X) / (n-1) for better numerical stability
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store explained variance
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
        
        # Determine number of components
        if self.n_components is None:
            n_components = self.n_features_
        elif isinstance(self.n_components, float):
            # Find number of components to explain desired variance
            cumsum = np.cumsum(self.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.n_components) + 1
        else:
            n_components = min(self.n_components, self.n_features_)
        
        # Select components
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]
        
        # Compute singular values (for compatibility with sklearn)
        self.singular_values_ = np.sqrt(self.explained_variance_ * (self.n_samples_ - 1))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to lower dimensional space.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        X = np.array(X, dtype=np.float64)
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project onto principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        # Apply whitening if requested
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_components)
            Transformed data
            
        Returns:
        --------
        X_original : np.ndarray of shape (n_samples, n_features)
            Data in original space
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        X = np.array(X, dtype=np.float64)
        
        # Undo whitening if it was applied
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_)
        
        # Project back to original space
        X_original = np.dot(X, self.components_) + self.mean_
        
        return X_original
    
    def score(self, X: np.ndarray) -> float:
        """
        Return the average log-likelihood of the data.
        
        This is a simplified version that returns the proportion of variance
        explained by the selected components.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        float
            Proportion of variance explained
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        return np.sum(self.explained_variance_ratio_)
    
    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the generative model.
        
        Returns:
        --------
        np.ndarray of shape (n_features, n_features)
            Estimated covariance matrix
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        # Covariance = components.T @ diag(explained_variance) @ components
        cov = self.components_.T @ np.diag(self.explained_variance_) @ self.components_
        
        return cov
    
    def plot_explained_variance(self, cumulative: bool = True) -> None:
        """
        Plot explained variance ratio.
        
        Parameters:
        -----------
        cumulative : bool, default=True
            Whether to plot cumulative variance
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        plt.figure(figsize=(10, 6))
        
        n_components = len(self.explained_variance_ratio_)
        x = np.arange(1, n_components + 1)
        
        if cumulative:
            plt.subplot(1, 2, 1)
            plt.plot(x, self.explained_variance_ratio_, 'bo-')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('Individual Explained Variance')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(x, np.cumsum(self.explained_variance_ratio_), 'ro-')
            plt.xlabel('Principal Component')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.title('Cumulative Explained Variance')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
        else:
            plt.plot(x, self.explained_variance_ratio_, 'bo-')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('Explained Variance per Component')
            plt.grid(True, alpha=0.3)
        
        plt.show()
    
    def plot_components(self, n_components: int = 4, 
                       feature_names: Optional[list] = None) -> None:
        """
        Plot the principal components.
        
        Parameters:
        -----------
        n_components : int, default=4
            Number of components to plot
        feature_names : list, optional
            Names of features
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        n_to_plot = min(n_components, self.components_.shape[0])
        
        plt.figure(figsize=(12, 3 * n_to_plot))
        
        for i in range(n_to_plot):
            plt.subplot(n_to_plot, 1, i + 1)
            
            if feature_names:
                x_labels = feature_names
                x_pos = np.arange(len(feature_names))
            else:
                x_labels = [f'Feature {j}' for j in range(self.n_features_)]
                x_pos = np.arange(self.n_features_)
            
            plt.bar(x_pos, self.components_[i])
            plt.title(f'Principal Component {i + 1} '
                     f'(Variance: {self.explained_variance_ratio_[i]:.3f})')
            plt.xlabel('Features')
            plt.ylabel('Component Weight')
            
            if len(x_labels) <= 20:  # Only show labels if not too many
                plt.xticks(x_pos, x_labels, rotation=45)
            
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_2d_projection(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          title: str = "PCA 2D Projection") -> None:
        """
        Plot 2D projection of the data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Original data
        y : np.ndarray of shape (n_samples,), optional
            Target labels for coloring
        title : str
            Plot title
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        if self.components_.shape[0] < 2:
            raise ValueError("Need at least 2 components for 2D projection")
        
        # Transform data
        X_transformed = self.transform(X)
        
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            unique_labels = np.unique(y)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = y == label
                plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1],
                           c=[colors[i]], label=f'Class {label}', alpha=0.7)
            plt.legend()
        else:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
        
        plt.xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()


def generate_sample_data(n_samples: int = 200, 
                        n_features: int = 10,
                        n_informative: int = 5,
                        noise: float = 0.1,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for PCA testing.
    
    Parameters:
    -----------
    n_samples : int, default=200
        Number of samples
    n_features : int, default=10
        Number of features
    n_informative : int, default=5
        Number of informative features
    noise : float, default=0.1
        Noise level
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    X : np.ndarray of shape (n_samples, n_features)
        Generated samples
    y : np.ndarray of shape (n_samples,)
        Generated labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate informative features
    X_informative = np.random.randn(n_samples, n_informative)
    
    # Create some correlation structure
    W = np.random.randn(n_informative, n_features)
    X = X_informative @ W
    
    # Add noise
    X += noise * np.random.randn(n_samples, n_features)
    
    # Generate labels based on first component
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("PCA Example")
    print("=" * 50)
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=200, n_features=20, 
                               n_informative=5, random_state=42)
    
    print(f"Original data shape: {X.shape}")
    
    # Test PCA with different numbers of components
    for n_components in [2, 5, 0.95]:
        print(f"\nPCA with {n_components} components:")
        
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)
        
        print(f"Transformed shape: {X_transformed.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
        print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # Test reconstruction
        X_reconstructed = pca.inverse_transform(X_transformed)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Visualizations
    pca_vis = PCA(n_components=5)
    pca_vis.fit(X)
    
    print("\nGenerating visualizations...")
    
    # Plot explained variance
    pca_vis.plot_explained_variance()
    
    # Plot components
    pca_vis.plot_components(n_components=3)
    
    # Plot 2D projection
    pca_vis.plot_2d_projection(X, y, "PCA 2D Projection with Class Labels")
