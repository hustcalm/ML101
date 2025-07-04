"""
Lasso Regression Implementation

Lasso (Least Absolute Shrinkage and Selection Operator) is a linear regression technique 
that includes L1 regularization. It performs both regularization and feature selection 
by shrinking some coefficients to exactly zero.

Mathematical Foundation:
- Cost Function: J(θ) = (1/2m) * Σ(hθ(x) - y)² + λ * Σ|θ|
- No closed-form solution due to non-differentiable L1 penalty
- Solved using coordinate descent or proximal gradient methods

Where λ (lambda) is the regularization parameter.

Author: ML101 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import warnings


class LassoRegression:
    """
    Lasso Regression implementation with L1 regularization.
    
    Lasso regression adds an L1 penalty term to the linear regression cost function,
    which performs both regularization and automatic feature selection by driving
    some coefficients to exactly zero.
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength (λ). Higher values mean stronger regularization.
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    max_iter : int, default=1000
        Maximum iterations for coordinate descent
    tolerance : float, default=1e-4
        Tolerance for convergence
    positive : bool, default=False
        Force coefficients to be positive
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tolerance: float = 1e-4,
                 positive: bool = False):
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.positive = positive
        
        # Model parameters
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.cost_history_ = []
        
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """
        Soft thresholding operator for L1 regularization.
        
        Parameters:
        -----------
        x : float
            Input value
        threshold : float
            Threshold value
            
        Returns:
        --------
        result : float
            Soft thresholded value
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Lasso regression cost function.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
            
        Returns:
        --------
        cost : float
            Lasso regression cost
        """
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        l1_penalty = self.alpha * np.sum(np.abs(self.coef_))
        return mse + l1_penalty
    
    def _coordinate_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using coordinate descent algorithm.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        if self.fit_intercept:
            self.intercept_ = np.mean(y)
        else:
            self.intercept_ = 0.0
        
        # Precompute X^T X diagonal and X^T y for efficiency
        XTX_diag = np.sum(X ** 2, axis=0)
        XTy = X.T @ y
        
        self.cost_history_ = []
        
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            # Update intercept if needed
            if self.fit_intercept:
                residuals = y - X @ self.coef_
                self.intercept_ = np.mean(residuals)
            
            # Coordinate descent: update each coefficient
            for j in range(n_features):
                # Compute residual without j-th feature
                residual_j = y - self.intercept_ - X @ self.coef_ + X[:, j] * self.coef_[j]
                
                # Compute optimal coefficient for j-th feature
                rho_j = X[:, j] @ residual_j
                
                if XTX_diag[j] != 0:
                    # Apply soft thresholding
                    if self.positive:
                        self.coef_[j] = max(0, (rho_j - self.alpha) / XTX_diag[j])
                    else:
                        self.coef_[j] = self._soft_threshold(rho_j / XTX_diag[j], 
                                                           self.alpha / XTX_diag[j])
                else:
                    self.coef_[j] = 0.0
            
            # Compute cost and check convergence
            cost = self._compute_cost(X, y)
            self.cost_history_.append(cost)
            
            # Check for convergence
            if np.max(np.abs(self.coef_ - coef_old)) < self.tolerance:
                self.n_iter_ = iteration + 1
                print(f"Converged after {self.n_iter_} iterations")
                break
        else:
            self.n_iter_ = self.max_iter
            warnings.warn(f"Maximum iterations ({self.max_iter}) reached. "
                         "Consider increasing max_iter or tolerance.")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegression':
        """
        Fit Lasso regression model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training targets
            
        Returns:
        --------
        self : LassoRegression
            Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Fit model using coordinate descent
        self._coordinate_descent(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values
        """
        if self.coef_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Parameters:
        -----------
        X : np.ndarray
            Test features
        y : np.ndarray
            Test targets
            
        Returns:
        --------
        r2_score : float
            R² coefficient of determination
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on absolute coefficient values.
        
        Returns:
        --------
        importance : np.ndarray
            Feature importance scores
        """
        if self.coef_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        importance = np.abs(self.coef_)
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def get_selected_features(self, threshold: float = 1e-6) -> np.ndarray:
        """
        Get indices of selected features (non-zero coefficients).
        
        Parameters:
        -----------
        threshold : float
            Threshold for considering a coefficient as non-zero
            
        Returns:
        --------
        selected : np.ndarray
            Indices of selected features
        """
        if self.coef_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        return np.where(np.abs(self.coef_) > threshold)[0]
    
    def plot_cost_history(self) -> None:
        """Plot cost function history."""
        if not self.cost_history_:
            print("No cost history available.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history_)
        plt.title('Lasso Regression Cost Function')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
    
    def plot_regularization_path(self, X: np.ndarray, y: np.ndarray, 
                                alphas: np.ndarray, feature_names: Optional[list] = None) -> None:
        """
        Plot coefficients as a function of regularization strength.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets
        alphas : np.ndarray
            Array of regularization strengths to try
        feature_names : list, optional
            Names of features for legend
        """
        coefficients = []
        
        for alpha in alphas:
            lasso = LassoRegression(alpha=alpha, max_iter=self.max_iter)
            lasso.fit(X, y)
            coefficients.append(lasso.coef_)
        
        coefficients = np.array(coefficients)
        
        plt.figure(figsize=(12, 8))
        for i in range(coefficients.shape[1]):
            label = feature_names[i] if feature_names else f'Feature {i+1}'
            plt.plot(alphas, coefficients[:, i], label=label)
        
        plt.xscale('log')
        plt.xlabel('Regularization Parameter (α)')
        plt.ylabel('Coefficient Value')
        plt.title('Lasso Regression: Regularization Path')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_selection(self, alphas: np.ndarray, n_features_selected: np.ndarray) -> None:
        """
        Plot number of selected features vs regularization strength.
        
        Parameters:
        -----------
        alphas : np.ndarray
            Array of regularization strengths
        n_features_selected : np.ndarray
            Number of features selected for each alpha
        """
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, n_features_selected, 'bo-')
        plt.xscale('log')
        plt.xlabel('Regularization Parameter (α)')
        plt.ylabel('Number of Selected Features')
        plt.title('Lasso Feature Selection')
        plt.grid(True)
        plt.show()


def generate_sparse_regression_data(n_samples: int = 100, n_features: int = 20, 
                                  n_informative: int = 5, noise: float = 0.1,
                                  random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data with sparse coefficients.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Total number of features
    n_informative : int
        Number of informative features
    noise : float
        Noise level
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    true_coef : np.ndarray
        True coefficients (sparse)
    """
    if random_state:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create sparse coefficient vector
    true_coef = np.zeros(n_features)
    informative_indices = np.random.choice(n_features, n_informative, replace=False)
    true_coef[informative_indices] = np.random.randn(n_informative) * 2
    
    # Generate target with noise
    y = X @ true_coef + noise * np.random.randn(n_samples)
    
    return X, y, true_coef


# Example usage and demonstration
if __name__ == "__main__":
    print("Lasso Regression Implementation Demo")
    print("=" * 40)
    
    # Generate synthetic sparse data
    X, y, true_coef = generate_sparse_regression_data(n_samples=100, n_features=20, 
                                                     n_informative=5, noise=0.1, 
                                                     random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True number of informative features: {np.sum(true_coef != 0)}")
    
    print("\n1. Lasso Regression with Different Regularization Strengths")
    
    # Compare different regularization strengths
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    for alpha in alphas:
        lasso = LassoRegression(alpha=alpha, max_iter=1000)
        lasso.fit(X_train, y_train)
        
        train_score = lasso.score(X_train, y_train)
        test_score = lasso.score(X_test, y_test)
        n_selected = len(lasso.get_selected_features())
        
        print(f"α = {alpha:6.3f}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}, "
              f"Features = {n_selected}")
    
    print("\n2. Feature Selection Analysis")
    
    # Optimal alpha (you would use cross-validation in practice)
    optimal_alpha = 0.1
    lasso = LassoRegression(alpha=optimal_alpha, max_iter=1000)
    lasso.fit(X_train, y_train)
    
    selected_features = lasso.get_selected_features()
    true_informative = np.where(true_coef != 0)[0]
    
    print(f"True informative features: {sorted(true_informative)}")
    print(f"Selected features: {sorted(selected_features)}")
    print(f"Correctly identified: {len(set(selected_features) & set(true_informative))}")
    
    print(f"\nTrue coefficients: {true_coef}")
    print(f"Lasso coefficients: {lasso.coef_}")
    
    print("\n3. Regularization Path Analysis")
    
    # Analyze regularization path
    alphas_path = np.logspace(-3, 1, 20)
    n_features_selected = []
    
    for alpha in alphas_path:
        lasso_temp = LassoRegression(alpha=alpha, max_iter=1000)
        lasso_temp.fit(X_train, y_train)
        n_features_selected.append(len(lasso_temp.get_selected_features()))
    
    # Visualization
    print("\n4. Generating Visualizations...")
    
    # Plot cost history
    lasso.plot_cost_history()
    
    # Plot regularization path
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    lasso.plot_regularization_path(X_train, y_train, alphas_path, feature_names)
    
    # Plot feature selection
    lasso.plot_feature_selection(alphas_path, n_features_selected)
    
    print("\n5. Coefficient Sparsity Comparison")
    
    # Compare sparsity levels
    alphas_sparse = [0.01, 0.1, 1.0]
    
    for alpha in alphas_sparse:
        lasso_temp = LassoRegression(alpha=alpha, max_iter=1000)
        lasso_temp.fit(X_train, y_train)
        
        n_nonzero = np.sum(np.abs(lasso_temp.coef_) > 1e-6)
        max_coef = np.max(np.abs(lasso_temp.coef_))
        
        print(f"α = {alpha:4.2f}: Non-zero coefficients = {n_nonzero:2d}, "
              f"Max |coefficient| = {max_coef:.4f}")
    
    print("\nDemo completed!")
