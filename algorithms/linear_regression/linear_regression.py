"""
Linear Regression Implementation from Scratch

This module implements Linear Regression using both analytical (Normal Equation) 
and iterative (Gradient Descent) methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class LinearRegression:
    """
    Linear Regression implementation from scratch.
    
    Supports both analytical solution (Normal Equation) and 
    iterative solution (Gradient Descent).
    
    Parameters:
    -----------
    method : str, default='normal'
        Method to use for fitting. Options: 'normal', 'gradient_descent'
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iterations : int, default=1000
        Maximum number of iterations for gradient descent
    tolerance : float, default=1e-6
        Convergence tolerance for gradient descent
    """
    
    def __init__(self, method='normal', learning_rate=0.01, 
                 max_iterations=1000, tolerance=1e-6):
        self.method = method
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : LinearRegression
            Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        
        # Add bias term (intercept)
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        if self.method == 'normal':
            self._fit_normal_equation(X_with_bias, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X_with_bias, y)
        else:
            raise ValueError("Method must be 'normal' or 'gradient_descent'")
            
        return self
    
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using the Normal Equation: θ = (X^T X)^(-1) X^T y
        """
        try:
            # Normal equation
            theta = np.linalg.inv(X.T @ X) @ X.T @ y
            self.bias = theta[0]
            self.weights = theta[1:] if len(theta) > 1 else np.array([theta[1]])
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            theta = np.linalg.pinv(X.T @ X) @ X.T @ y
            self.bias = theta[0]
            self.weights = theta[1:] if len(theta) > 1 else np.array([theta[1]])
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using Gradient Descent.
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        theta = np.zeros(n_features)
        
        self.cost_history = []
        
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = X @ theta
            
            # Compute cost (Mean Squared Error)
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = (2 / n_samples) * X.T @ (y_pred - y)
            
            # Update parameters
            theta_new = theta - self.learning_rate * gradients
            
            # Check for convergence
            if np.allclose(theta, theta_new, atol=self.tolerance):
                print(f"Converged after {i+1} iterations")
                break
                
            theta = theta_new
        
        self.bias = theta[0]
        self.weights = theta[1:] if len(theta) > 1 else np.array([theta[1]])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values
        """
        if self.weights is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True target values
            
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
        """Plot the cost function history (only for gradient descent)."""
        if not self.cost_history:
            print("No cost history available. Use gradient descent method.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()


def generate_linear_data(n_samples: int = 100, noise: float = 0.1, 
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Standard deviation of Gaussian noise
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X, y : tuple of np.ndarray
        Features and target values
    """
    if random_state:
        np.random.seed(random_state)
    
    X = np.random.uniform(-3, 3, (n_samples, 1))
    true_weights = np.array([2.5])
    true_bias = 1.0
    y = X @ true_weights + true_bias + np.random.normal(0, noise, n_samples)
    
    return X, y.ravel()
