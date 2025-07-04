"""
Logistic Regression Implementation from Scratch

This module implements Logistic Regression for binary and multiclass classification
using gradient descent optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union


class LogisticRegression:
    """
    Logistic Regression implementation from scratch.
    
    Supports binary and multiclass classification using One-vs-Rest approach.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iterations : int, default=1000
        Maximum number of iterations
    tolerance : float, default=1e-6
        Convergence tolerance
    regularization : str, default=None
        Type of regularization ('l1', 'l2', or None)
    lambda_reg : float, default=0.01
        Regularization strength
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, 
                 tolerance=1e-6, regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.classes_ = None
        self.n_classes = None
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with numerical stability.
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function for multiclass classification.
        """
        # Subtract max for numerical stability
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _add_regularization(self, gradient: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Add regularization to gradient.
        """
        if self.regularization == 'l1':
            return gradient + self.lambda_reg * np.sign(weights)
        elif self.regularization == 'l2':
            return gradient + self.lambda_reg * weights
        return gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : LogisticRegression
            Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        if self.n_classes == 2:
            self._fit_binary(X, y)
        else:
            self._fit_multiclass(X, y)
            
        return self
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit binary logistic regression.
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Convert labels to 0 and 1
        y_binary = (y == self.classes_[1]).astype(int)
        
        for i in range(self.max_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)
            
            # Compute cost (logistic loss)
            cost = self._compute_binary_cost(y_binary, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y_binary)
            db = (1 / n_samples) * np.sum(y_pred - y_binary)
            
            # Add regularization
            if self.regularization:
                dw = self._add_regularization(dw, self.weights)
            
            # Update parameters
            weights_new = self.weights - self.learning_rate * dw
            bias_new = self.bias - self.learning_rate * db
            
            # Check convergence
            if (np.allclose(self.weights, weights_new, atol=self.tolerance) and
                abs(self.bias - bias_new) < self.tolerance):
                print(f"Converged after {i+1} iterations")
                break
                
            self.weights = weights_new
            self.bias = bias_new
    
    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit multiclass logistic regression using One-vs-Rest.
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters for each class
        self.weights = np.zeros((self.n_classes, n_features))
        self.bias = np.zeros(self.n_classes)
        self.cost_history = []
        
        # One-vs-Rest training
        for class_idx, class_label in enumerate(self.classes_):
            print(f"Training classifier for class {class_label}")
            
            # Create binary labels for current class
            y_binary = (y == class_label).astype(int)
            
            # Initialize parameters for this classifier
            weights = np.zeros(n_features)
            bias = 0
            
            for i in range(self.max_iterations):
                # Forward pass
                z = X @ weights + bias
                y_pred = self._sigmoid(z)
                
                # Compute gradients
                dw = (1 / n_samples) * X.T @ (y_pred - y_binary)
                db = (1 / n_samples) * np.sum(y_pred - y_binary)
                
                # Add regularization
                if self.regularization:
                    dw = self._add_regularization(dw, weights)
                
                # Update parameters
                weights_new = weights - self.learning_rate * dw
                bias_new = bias - self.learning_rate * db
                
                # Check convergence
                if (np.allclose(weights, weights_new, atol=self.tolerance) and
                    abs(bias - bias_new) < self.tolerance):
                    break
                    
                weights = weights_new
                bias = bias_new
            
            # Store parameters for this class
            self.weights[class_idx] = weights
            self.bias[class_idx] = bias
    
    def _compute_binary_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary logistic regression cost.
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Logistic loss
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add regularization cost
        if self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            cost += self.lambda_reg * np.sum(self.weights ** 2) / 2
            
        return cost
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        probabilities : np.ndarray
            Predicted probabilities for each class
        """
        if self.weights is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        
        if self.n_classes == 2:
            # Binary classification
            z = X @ self.weights + self.bias
            prob_positive = self._sigmoid(z)
            return np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Multiclass classification
            scores = X @ self.weights.T + self.bias
            return self._softmax(scores)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True target values
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_cost_history(self) -> None:
        """Plot the cost function history."""
        if not self.cost_history:
            print("No cost history available.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Logistic Loss')
        plt.grid(True)
        plt.show()


def generate_classification_data(n_samples: int = 200, n_features: int = 2, 
                               n_classes: int = 2, noise: float = 0.1,
                               random_state: Optional[int] = None) -> tuple:
    """
    Generate synthetic classification data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    noise : float
        Noise level
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    X, y : tuple
        Features and labels
    """
    if random_state:
        np.random.seed(random_state)
    
    # Generate class centers
    centers = np.random.randn(n_classes, n_features) * 3
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        # Generate samples around each center
        class_samples = np.random.randn(samples_per_class, n_features) * noise + centers[i]
        X.append(class_samples)
        y.extend([i] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle the data
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]
