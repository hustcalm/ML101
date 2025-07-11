"""
Support Vector Machine (SVM) Implementation

This module implements Support Vector Machines for both classification and regression
using Sequential Minimal Optimization (SMO) algorithm for solving the quadratic
programming problem.

Mathematical Foundation:
- SVM finds the optimal hyperplane that separates classes with maximum margin
- The decision function is: f(x) = sign(Σ αᵢ yᵢ K(xᵢ, x) + b)
- Where αᵢ are Lagrange multipliers, yᵢ are labels, K is kernel function, b is bias
- Support vectors are training points with αᵢ > 0

Author: ML101 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Callable, Tuple
import warnings


class SVM:
    """
    Support Vector Machine classifier using Sequential Minimal Optimization (SMO).
    
    This implementation supports both linear and non-linear classification through
    various kernel functions including RBF, polynomial, and sigmoid kernels.
    """
    
    def __init__(self, 
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: Union[str, float] = 'scale',
                 coef0: float = 0.0,
                 tolerance: float = 1e-3,
                 max_iter: int = 1000,
                 random_state: Optional[int] = None):
        """
        Initialize SVM classifier.
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter. Higher C means less regularization.
        kernel : str, default='rbf'
            Kernel function: 'linear', 'rbf', 'poly', 'sigmoid'
        degree : int, default=3
            Degree of polynomial kernel
        gamma : str or float, default='scale'
            Kernel coefficient for rbf/poly/sigmoid
        coef0 : float, default=0.0
            Independent term in kernel function
        tolerance : float, default=1e-3
            Tolerance for stopping criterion
        max_iter : int, default=1000
            Maximum number of iterations
        random_state : int, optional
            Random seed for reproducibility
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize attributes
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.n_support_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel function between two sets of vectors.
        
        Parameters:
        -----------
        X1, X2 : np.ndarray
            Input vectors
            
        Returns:
        --------
        np.ndarray
            Kernel matrix
        """
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'rbf':
            # Compute gamma if set to 'scale'
            if self.gamma == 'scale':
                gamma = 1.0 / (X1.shape[1] * X1.var())
            else:
                gamma = self.gamma
            
            # RBF kernel: K(x1, x2) = exp(-γ||x1-x2||²)
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                      np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * sq_dists)
        
        elif self.kernel == 'poly':
            # Polynomial kernel: K(x1, x2) = (γ<x1,x2> + r)^d
            if self.gamma == 'scale':
                gamma = 1.0 / X1.shape[1]
            else:
                gamma = self.gamma
            return (gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'sigmoid':
            # Sigmoid kernel: K(x1, x2) = tanh(γ<x1,x2> + r)
            if self.gamma == 'scale':
                gamma = 1.0 / X1.shape[1]
            else:
                gamma = self.gamma
            return np.tanh(gamma * np.dot(X1, X2.T) + self.coef0)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.
        
        Parameters:
        -----------
        X : np.ndarray
            Input samples
            
        Returns:
        --------
        np.ndarray
            Decision function values
        """
        if self.support_vectors_ is None:
            raise ValueError("Model not fitted yet")
        
        K = self._kernel_function(X, self.support_vectors_)
        decision = np.dot(K, self.dual_coef_) + self.intercept_
        return decision.flatten()
    
    def _smo_algorithm(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Sequential Minimal Optimization algorithm for solving SVM dual problem.
        
        This is a simplified version of SMO that handles the main optimization steps.
        """
        n_samples = X.shape[0]
        
        # Initialize alpha coefficients
        alpha = np.zeros(n_samples)
        b = 0.0
        
        # Precompute kernel matrix
        K = self._kernel_function(X, X)
        
        # Main SMO loop
        for iteration in range(self.max_iter):
            num_changed_alphas = 0
            
            for i in range(n_samples):
                # Calculate error for sample i
                decision_i = np.sum(alpha * y * K[i, :]) + b
                E_i = decision_i - y[i]
                
                # Check KKT conditions
                if ((y[i] * E_i < -self.tolerance and alpha[i] < self.C) or
                    (y[i] * E_i > self.tolerance and alpha[i] > 0)):
                    
                    # Select second alpha randomly
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Calculate error for sample j
                    decision_j = np.sum(alpha * y * K[j, :]) + b
                    E_j = decision_j - y[j]
                    
                    # Save old alphas
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    alpha[j] = alpha[j] - y[j] * (E_i - E_j) / eta
                    
                    # Clip alpha_j
                    if alpha[j] > H:
                        alpha[j] = H
                    elif alpha[j] < L:
                        alpha[j] = L
                    
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
                    
                    # Update bias
                    b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                break
        
        # Store support vectors
        support_mask = alpha > 1e-8
        self.support_vectors_ = X[support_mask]
        self.support_vector_labels_ = y[support_mask]
        self.dual_coef_ = (alpha * y)[support_mask]
        self.intercept_ = b
        self.n_support_ = np.sum(support_mask)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Fit the SVM model to training data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values (must be -1 or 1)
            
        Returns:
        --------
        self : SVM
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        # Validate input
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        # Convert labels to -1, 1 format
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM supports only binary classification")
        
        self.classes_ = unique_labels
        y_binary = np.where(y == unique_labels[0], -1, 1)
        
        # Run SMO algorithm
        self._smo_algorithm(X, y_binary)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        decision_values = self._decision_function(X)
        predictions = np.where(decision_values >= 0, 1, -1)
        
        # Convert back to original labels
        return np.where(predictions == -1, self.classes_[0], self.classes_[1])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Note: This uses Platt scaling approximation for probability estimates.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        np.ndarray of shape (n_samples, 2)
            Predicted class probabilities
        """
        X = np.array(X)
        decision_values = self._decision_function(X)
        
        # Simple sigmoid approximation (Platt scaling would be more accurate)
        proba_positive = 1 / (1 + np.exp(-decision_values))
        proba_negative = 1 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples
        y : np.ndarray of shape (n_samples,)
            True labels
            
        Returns:
        --------
        float
            Mean accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                              title: str = "SVM Decision Boundary") -> None:
        """
        Plot decision boundary for 2D data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, 2)
            Input samples (must be 2D)
        y : np.ndarray of shape (n_samples,)
            True labels
        title : str
            Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Can only plot decision boundary for 2D data")
        
        plt.figure(figsize=(10, 8))
        
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self._decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, 
                   colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])
        
        # Plot data points
        unique_labels = np.unique(y)
        colors = ['red', 'blue']
        for i, label in enumerate(unique_labels):
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                       label=f'Class {label}', alpha=0.7)
        
        # Plot support vectors
        if self.support_vectors_ is not None:
            plt.scatter(self.support_vectors_[:, 0], self.support_vectors_[:, 1], 
                       s=100, facecolors='none', edgecolors='black', linewidth=2,
                       label='Support Vectors')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def generate_classification_data(n_samples: int = 100, 
                               n_features: int = 2,
                               n_classes: int = 2,
                               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data for testing SVM.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples
    n_features : int, default=2
        Number of features
    n_classes : int, default=2
        Number of classes
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
    
    # Generate random centers for each class
    centers = np.random.randn(n_classes, n_features) * 3
    
    # Generate samples for each class
    samples_per_class = n_samples // n_classes
    X = []
    y = []
    
    for i in range(n_classes):
        # Generate samples around center
        class_samples = np.random.randn(samples_per_class, n_features) + centers[i]
        X.append(class_samples)
        y.append(np.full(samples_per_class, i))
    
    # Handle remaining samples
    remaining = n_samples - samples_per_class * n_classes
    if remaining > 0:
        class_idx = np.random.randint(0, n_classes)
        extra_samples = np.random.randn(remaining, n_features) + centers[class_idx]
        X.append(extra_samples)
        y.append(np.full(remaining, class_idx))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


if __name__ == "__main__":
    # Example usage
    print("SVM Example")
    print("=" * 50)
    
    # Generate sample data
    X, y = generate_classification_data(n_samples=200, random_state=42)
    
    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test different kernels
    kernels = ['linear', 'rbf', 'poly']
    
    for kernel in kernels:
        print(f"\nTesting {kernel.upper()} kernel:")
        svm = SVM(kernel=kernel, C=1.0, random_state=42)
        svm.fit(X_train, y_train)
        
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print(f"Number of support vectors: {svm.n_support_}")
        
        # Plot decision boundary for 2D data
        if X.shape[1] == 2:
            svm.plot_decision_boundary(X_train, y_train, 
                                     title=f"SVM Decision Boundary ({kernel.upper()} kernel)")
