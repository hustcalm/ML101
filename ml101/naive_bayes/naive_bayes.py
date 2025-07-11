"""
Naive Bayes Implementation from Scratch

This module implements different variants of Naive Bayes classifiers:
- Gaussian Naive Bayes (for continuous features)
- Multinomial Naive Bayes (for discrete features)
- Bernoulli Naive Bayes (for binary features)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Literal
from collections import defaultdict
import math


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.
    
    Assumes that continuous features follow a Gaussian distribution
    and features are conditionally independent given the class.
    
    Parameters:
    -----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_priors_ = None
        self.feature_means_ = None
        self.feature_vars_ = None
        self.n_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Fit the Gaussian Naive Bayes classifier.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : GaussianNaiveBayes
            Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize parameters
        self.class_priors_ = np.zeros(n_classes)
        self.feature_means_ = np.zeros((n_classes, self.n_features_))
        self.feature_vars_ = np.zeros((n_classes, self.n_features_))
        
        # Calculate parameters for each class
        for i, class_label in enumerate(self.classes_):
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            # Prior probability
            self.class_priors_[i] = np.sum(class_mask) / len(y)
            
            # Mean and variance for each feature
            self.feature_means_[i, :] = np.mean(X_class, axis=0)
            self.feature_vars_[i, :] = np.var(X_class, axis=0) + self.var_smoothing
        
        return self
    
    def _calculate_likelihood(self, X: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Calculate likelihood P(X|y) for a given class.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        class_idx : int
            Index of the class
            
        Returns:
        --------
        likelihood : np.ndarray
            Likelihood values for each sample
        """
        means = self.feature_means_[class_idx]
        vars = self.feature_vars_[class_idx]
        
        # Calculate Gaussian probability density
        # P(x|y) = (1/√(2πσ²)) * exp(-((x-μ)²)/(2σ²))
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * vars))
        log_likelihood -= 0.5 * np.sum(((X - means) ** 2) / vars, axis=1)
        
        return log_likelihood
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        if self.classes_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Calculate log posterior for each class
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            log_prior = np.log(self.class_priors_[i])
            log_likelihood = self._calculate_likelihood(X, i)
            log_posteriors[:, i] = log_prior + log_likelihood
        
        # Convert to probabilities using softmax to avoid numerical issues
        # P(y|X) = exp(log_posterior) / sum(exp(log_posterior))
        max_log_posterior = np.max(log_posteriors, axis=1, keepdims=True)
        exp_log_posteriors = np.exp(log_posteriors - max_log_posterior)
        probabilities = exp_log_posteriors / np.sum(exp_log_posteriors, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
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
            True labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier.
    
    Suitable for discrete features (e.g., word counts in text classification).
    Features are assumed to be generated from a multinomial distribution.
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = None
        self.feature_log_probs_ = None
        self.n_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNaiveBayes':
        """
        Fit the Multinomial Naive Bayes classifier.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (should be non-negative)
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : MultinomialNaiveBayes
            Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        
        if np.any(X < 0):
            raise ValueError("Multinomial Naive Bayes requires non-negative features")
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize parameters
        self.class_priors_ = np.zeros(n_classes)
        self.feature_log_probs_ = np.zeros((n_classes, self.n_features_))
        
        # Calculate parameters for each class
        for i, class_label in enumerate(self.classes_):
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            # Prior probability
            self.class_priors_[i] = np.sum(class_mask) / len(y)
            
            # Feature probabilities with Laplace smoothing
            feature_counts = np.sum(X_class, axis=0)  # Sum of feature counts for this class
            total_count = np.sum(feature_counts)      # Total count for this class
            
            # P(feature|class) = (count + alpha) / (total_count + alpha * n_features)
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = total_count + self.alpha * self.n_features_
            
            self.feature_log_probs_[i, :] = np.log(smoothed_counts / smoothed_total)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        if self.classes_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Calculate log posterior for each class
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            log_prior = np.log(self.class_priors_[i])
            # For multinomial: P(X|y) = ∏ P(xi|y)^count_i
            log_likelihood = np.sum(X * self.feature_log_probs_[i, :], axis=1)
            log_posteriors[:, i] = log_prior + log_likelihood
        
        # Convert to probabilities
        max_log_posterior = np.max(log_posteriors, axis=1, keepdims=True)
        exp_log_posteriors = np.exp(log_posteriors - max_log_posterior)
        probabilities = exp_log_posteriors / np.sum(exp_log_posteriors, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
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
            True labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes classifier.
    
    Suitable for binary features. Features are assumed to be generated
    from a Bernoulli distribution.
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
    binarize : float, default=0.0
        Threshold for binarizing features. If None, features are assumed
        to be already binary.
    """
    
    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0):
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_priors_ = None
        self.feature_probs_ = None
        self.n_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BernoulliNaiveBayes':
        """
        Fit the Bernoulli Naive Bayes classifier.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : BernoulliNaiveBayes
            Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        
        # Binarize features if threshold is specified
        if self.binarize is not None:
            X = (X > self.binarize).astype(int)
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize parameters
        self.class_priors_ = np.zeros(n_classes)
        self.feature_probs_ = np.zeros((n_classes, self.n_features_))
        
        # Calculate parameters for each class
        for i, class_label in enumerate(self.classes_):
            class_mask = (y == class_label)
            X_class = X[class_mask]
            n_samples_class = np.sum(class_mask)
            
            # Prior probability
            self.class_priors_[i] = n_samples_class / len(y)
            
            # Feature probabilities with Laplace smoothing
            # P(feature=1|class) = (count_1 + alpha) / (total_count + 2*alpha)
            feature_counts = np.sum(X_class, axis=0)
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = n_samples_class + 2 * self.alpha
            
            self.feature_probs_[i, :] = smoothed_counts / smoothed_total
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        if self.classes_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        
        # Binarize features if threshold is specified
        if self.binarize is not None:
            X = (X > self.binarize).astype(int)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Calculate log posterior for each class
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            log_prior = np.log(self.class_priors_[i])
            
            # For Bernoulli: P(X|y) = ∏ P(xi|y)^xi * (1-P(xi|y))^(1-xi)
            # Log likelihood = Σ [xi * log(P(xi|y)) + (1-xi) * log(1-P(xi|y))]
            prob_1 = self.feature_probs_[i, :]
            prob_0 = 1 - prob_1
            
            log_likelihood = np.sum(
                X * np.log(prob_1 + 1e-10) + (1 - X) * np.log(prob_0 + 1e-10),
                axis=1
            )
            
            log_posteriors[:, i] = log_prior + log_likelihood
        
        # Convert to probabilities
        max_log_posterior = np.max(log_posteriors, axis=1, keepdims=True)
        exp_log_posteriors = np.exp(log_posteriors - max_log_posterior)
        probabilities = exp_log_posteriors / np.sum(exp_log_posteriors, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
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
            True labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def generate_naive_bayes_data(variant: str = 'gaussian', n_samples: int = 200,
                             n_features: int = 2, n_classes: int = 2,
                             random_state: Optional[int] = None) -> tuple:
    """
    Generate synthetic data suitable for Naive Bayes classifiers.
    
    Parameters:
    -----------
    variant : str
        Type of data: 'gaussian', 'multinomial', or 'bernoulli'
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    X, y : tuple
        Features and labels
    """
    if random_state:
        np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    
    X = []
    y = []
    
    for class_idx in range(n_classes):
        if variant == 'gaussian':
            # Generate Gaussian data with different means for each class
            mean = np.random.randn(n_features) * 2
            cov = np.eye(n_features) * 0.5
            X_class = np.random.multivariate_normal(mean, cov, samples_per_class)
        
        elif variant == 'multinomial':
            # Generate multinomial data (word counts)
            # Different classes have different word preferences
            prob_dist = np.random.dirichlet(np.ones(n_features) * (class_idx + 1))
            X_class = np.random.multinomial(20, prob_dist, samples_per_class)
        
        elif variant == 'bernoulli':
            # Generate binary data
            prob_1 = 0.3 + 0.4 * class_idx / (n_classes - 1)  # Different probability for each class
            X_class = np.random.binomial(1, prob_1, (samples_per_class, n_features))
        
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        X.append(X_class)
        y.extend([class_idx] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle the data
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


class NaiveBayesVisualization:
    """
    Visualization utilities for Naive Bayes classifiers.
    """
    
    @staticmethod
    def plot_gaussian_distributions(nb_model: GaussianNaiveBayes, feature_idx: int = 0,
                                   feature_name: str = "Feature") -> None:
        """
        Plot Gaussian distributions for each class.
        
        Parameters:
        -----------
        nb_model : GaussianNaiveBayes
            Fitted Gaussian Naive Bayes model
        feature_idx : int
            Index of feature to plot
        feature_name : str
            Name of the feature
        """
        if nb_model.classes_ is None:
            print("Model hasn't been fitted yet.")
            return
        
        plt.figure(figsize=(10, 6))
        
        x_range = np.linspace(
            np.min(nb_model.feature_means_[:, feature_idx]) - 3 * np.sqrt(np.max(nb_model.feature_vars_[:, feature_idx])),
            np.max(nb_model.feature_means_[:, feature_idx]) + 3 * np.sqrt(np.max(nb_model.feature_vars_[:, feature_idx])),
            1000
        )
        
        for i, class_label in enumerate(nb_model.classes_):
            mean = nb_model.feature_means_[i, feature_idx]
            var = nb_model.feature_vars_[i, feature_idx]
            
            # Calculate Gaussian probability density
            pdf = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x_range - mean) ** 2) / var)
            
            plt.plot(x_range, pdf, label=f'Class {class_label}', linewidth=2)
            plt.axvline(mean, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=0.7)
        
        plt.xlabel(feature_name)
        plt.ylabel('Probability Density')
        plt.title(f'Gaussian Distributions for {feature_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(nb_model, X: np.ndarray, y: np.ndarray,
                             title: str = "Naive Bayes Decision Boundary") -> None:
        """
        Plot decision boundary for 2D data.
        
        Parameters:
        -----------
        nb_model : NaiveBayes
            Fitted Naive Bayes model
        X : np.ndarray
            Training features (2D)
        y : np.ndarray
            Training labels
        title : str
            Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Can only plot decision boundary for 2D data")
        
        # Create a mesh
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = nb_model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    
    @staticmethod
    def plot_feature_probabilities(nb_model, feature_names: Optional[List[str]] = None) -> None:
        """
        Plot feature probabilities for each class (for Multinomial/Bernoulli NB).
        
        Parameters:
        -----------
        nb_model : MultinomialNaiveBayes or BernoulliNaiveBayes
            Fitted Naive Bayes model
        feature_names : List[str], optional
            Names of features
        """
        if hasattr(nb_model, 'feature_log_probs_'):
            probs = np.exp(nb_model.feature_log_probs_)
        elif hasattr(nb_model, 'feature_probs_'):
            probs = nb_model.feature_probs_
        else:
            print("Model doesn't have feature probabilities.")
            return
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(probs.shape[1])]
        
        n_classes = len(nb_model.classes_)
        
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6))
        if n_classes == 1:
            axes = [axes]
        
        for i, class_label in enumerate(nb_model.classes_):
            axes[i].bar(range(len(feature_names)), probs[i, :])
            axes[i].set_title(f'Class {class_label}')
            axes[i].set_xlabel('Features')
            axes[i].set_ylabel('Probability')
            axes[i].set_xticks(range(len(feature_names)))
            axes[i].set_xticklabels(feature_names, rotation=45)
        
        plt.tight_layout()
        plt.show()
