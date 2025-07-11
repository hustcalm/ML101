"""
Ridge Regression Implementation

Ridge Regression is a linear regression technique that includes L2 regularization to prevent overfitting.
It adds a penalty term proportional to the square of the coefficients to the ordinary least squares cost function.

Mathematical Foundation:
- Cost Function: J(θ) = (1/2m) * Σ(hθ(x) - y)² + λ * Σθ²
- Normal Equation: θ = (X^T X + λI)^(-1) X^T y
- Gradient Descent: θ := θ - α * [(1/m) * X^T(Xθ - y) + λθ]

Where λ (lambda) is the regularization parameter.

Author: ML101 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import warnings


class RidgeRegression:
    """
    Ridge Regression implementation with L2 regularization.
    
    Ridge regression adds a penalty term to the linear regression cost function
    to prevent overfitting by shrinking the coefficients.
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength (λ). Higher values mean stronger regularization.
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    solver : str, default='normal'
        Solver to use: 'normal' (closed-form) or 'gradient_descent'
    max_iter : int, default=1000
        Maximum iterations for gradient descent
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    tolerance : float, default=1e-6
        Tolerance for convergence in gradient descent
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 solver: str = 'normal',
                 max_iter: int = 1000,
                 learning_rate: float = 0.01,
                 tolerance: float = 1e-6):
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        
        # Model parameters
        self.coef_ = None
        self.intercept_ = None
        self.cost_history_ = []
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept term to feature matrix."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute Ridge regression cost function.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (with intercept if fit_intercept=True)
        y : np.ndarray
            Target vector
        theta : np.ndarray
            Parameter vector
            
        Returns:
        --------
        cost : float
            Ridge regression cost
        """
        m = X.shape[0]
        predictions = X @ theta
        mse = np.mean((predictions - y) ** 2)
        
        # L2 regularization term (don't regularize intercept)
        if self.fit_intercept:
            l2_penalty = self.alpha * np.sum(theta[1:] ** 2)
        else:
            l2_penalty = self.alpha * np.sum(theta ** 2)
            
        return mse + l2_penalty
    
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit using normal equation (closed-form solution).
        
        θ = (X^T X + λI)^(-1) X^T y
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # Create regularization matrix (don't regularize intercept)
        reg_matrix = self.alpha * np.eye(X.shape[1])
        if self.fit_intercept:
            reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        try:
            # Normal equation with regularization
            theta = np.linalg.inv(X.T @ X + reg_matrix) @ X.T @ y
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            warnings.warn("Matrix is singular, using pseudo-inverse")
            theta = np.linalg.pinv(X.T @ X + reg_matrix) @ X.T @ y
        
        return theta
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit using gradient descent.
        
        Gradient: ∇J = (1/m) * X^T(Xθ - y) + λθ
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        m, n = X.shape
        theta = np.random.normal(0, 0.01, n)
        
        self.cost_history_ = []
        
        for i in range(self.max_iter):
            # Compute predictions and cost
            predictions = X @ theta
            cost = self._compute_cost(X, y, theta)
            self.cost_history_.append(cost)
            
            # Compute gradients
            gradients = (1/m) * X.T @ (predictions - y)
            
            # Add L2 regularization to gradients (don't regularize intercept)
            if self.fit_intercept:
                gradients[1:] += self.alpha * theta[1:]
            else:
                gradients += self.alpha * theta
            
            # Update parameters
            theta -= self.learning_rate * gradients
            
            # Check for convergence
            if i > 0 and abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
        
        return theta
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """
        Fit Ridge regression model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training targets
            
        Returns:
        --------
        self : RidgeRegression
            Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Fit model using specified solver
        if self.solver == 'normal':
            theta = self._fit_normal_equation(X, y)
        elif self.solver == 'gradient_descent':
            theta = self._fit_gradient_descent(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        # Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        
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
    
    def plot_cost_history(self) -> None:
        """Plot cost function history (only for gradient descent)."""
        if not self.cost_history_:
            print("No cost history available. Use solver='gradient_descent' to track cost.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history_)
        plt.title('Ridge Regression Cost Function')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
    
    def plot_regularization_path(self, X: np.ndarray, y: np.ndarray, 
                                alphas: np.ndarray) -> None:
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
        """
        coefficients = []
        
        for alpha in alphas:
            ridge = RidgeRegression(alpha=alpha, solver='normal')
            ridge.fit(X, y)
            coefficients.append(ridge.coef_)
        
        coefficients = np.array(coefficients)
        
        plt.figure(figsize=(12, 8))
        for i in range(coefficients.shape[1]):
            plt.plot(alphas, coefficients[:, i], label=f'Feature {i+1}')
        
        plt.xscale('log')
        plt.xlabel('Regularization Parameter (α)')
        plt.ylabel('Coefficient Value')
        plt.title('Ridge Regression: Regularization Path')
        plt.legend()
        plt.grid(True)
        plt.show()


def generate_regression_data(n_samples: int = 100, n_features: int = 5, 
                           noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data with multicollinearity.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
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
    """
    if random_state:
        np.random.seed(random_state)
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Add multicollinearity
    if n_features >= 3:
        X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
        X[:, 2] = X[:, 0] - 0.3 * X[:, 1] + 0.3 * np.random.randn(n_samples)
    
    # True coefficients
    true_coef = np.random.randn(n_features)
    
    # Generate target with noise
    y = X @ true_coef + noise * np.random.randn(n_samples)
    
    return X, y


# Example usage and demonstration
if __name__ == "__main__":
    print("Ridge Regression Implementation Demo")
    print("=" * 40)
    
    # Generate synthetic data
    X, y = generate_regression_data(n_samples=100, n_features=5, noise=0.1, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print("\n1. Ridge Regression vs Linear Regression Comparison")
    
    # Compare different regularization strengths
    alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
    
    for alpha in alphas:
        ridge = RidgeRegression(alpha=alpha, solver='normal')
        ridge.fit(X_train, y_train)
        
        train_score = ridge.score(X_train, y_train)
        test_score = ridge.score(X_test, y_test)
        
        print(f"α = {alpha:6.1f}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
    
    print("\n2. Gradient Descent vs Normal Equation")
    
    # Compare solvers
    ridge_normal = RidgeRegression(alpha=1.0, solver='normal')
    ridge_gd = RidgeRegression(alpha=1.0, solver='gradient_descent', 
                              learning_rate=0.01, max_iter=1000)
    
    ridge_normal.fit(X_train, y_train)
    ridge_gd.fit(X_train, y_train)
    
    print(f"Normal Equation - Train R²: {ridge_normal.score(X_train, y_train):.4f}")
    print(f"Normal Equation - Test R²: {ridge_normal.score(X_test, y_test):.4f}")
    print(f"Gradient Descent - Train R²: {ridge_gd.score(X_train, y_train):.4f}")
    print(f"Gradient Descent - Test R²: {ridge_gd.score(X_test, y_test):.4f}")
    
    print("\n3. Coefficient Comparison")
    print(f"Normal Equation Coefficients: {ridge_normal.coef_}")
    print(f"Gradient Descent Coefficients: {ridge_gd.coef_}")
    
    # Visualization
    print("\n4. Generating Visualizations...")
    
    # Plot cost history for gradient descent
    ridge_gd.plot_cost_history()
    
    # Plot regularization path
    alphas_path = np.logspace(-3, 3, 50)
    ridge_normal.plot_regularization_path(X_train, y_train, alphas_path)
    
    print("\nDemo completed!")
