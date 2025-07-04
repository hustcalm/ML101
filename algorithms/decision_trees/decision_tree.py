"""
Decision Tree Implementation from Scratch

This module implements Decision Trees for both classification and regression
using various splitting criteria and pruning techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Literal, Dict, Any
from collections import Counter
import math


class DecisionTreeNode:
    """
    Node class for Decision Tree.
    """
    
    def __init__(self, feature_index: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['DecisionTreeNode'] = None, right: Optional['DecisionTreeNode'] = None,
                 value: Optional[Union[float, int]] = None, samples: int = 0, 
                 class_counts: Optional[dict] = None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value for split
        self.left = left                   # Left child node
        self.right = right                 # Right child node
        self.value = value                 # Prediction value (for leaf nodes)
        self.samples = samples             # Number of samples in this node
        self.class_counts = class_counts   # Class counts for classification


class DecisionTree:
    """
    Decision Tree implementation from scratch.
    
    Supports both classification and regression with various splitting criteria.
    
    Parameters:
    -----------
    task : str, default='classification'
        Task type: 'classification' or 'regression'
    criterion : str, default='gini'
        Splitting criterion: 'gini', 'entropy' (classification) or 'mse' (regression)
    max_depth : int, default=None
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
    max_features : int or str, default=None
        Number of features to consider when looking for the best split
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, task: Literal['classification', 'regression'] = 'classification',
                 criterion: str = 'gini', max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, str]] = None,
                 random_state: Optional[int] = None):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        
        # Set default criterion based on task
        if task == 'classification' and criterion not in ['gini', 'entropy']:
            self.criterion = 'gini'
        elif task == 'regression' and criterion not in ['mse', 'mae']:
            self.criterion = 'mse'
    
    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy."""
        if len(y) == 0:
            return 0
        
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _calculate_mse(self, y: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        if len(y) == 0:
            return 0
        
        mean_y = np.mean(y)
        mse = np.mean((y - mean_y) ** 2)
        return mse
    
    def _calculate_mae(self, y: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        if len(y) == 0:
            return 0
        
        median_y = np.median(y)
        mae = np.mean(np.abs(y - median_y))
        return mae
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        elif self.criterion == 'entropy':
            return self._calculate_entropy(y)
        elif self.criterion == 'mse':
            return self._calculate_mse(y)
        elif self.criterion == 'mae':
            return self._calculate_mae(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _get_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best split for the given data.
        
        Returns:
        --------
        best_feature : int
            Index of the best feature to split on
        best_threshold : float
            Best threshold value
        best_gain : float
            Information gain from the split
        """
        n_samples, n_features = X.shape
        
        # Determine features to consider
        if self.max_features is None:
            features_to_consider = range(n_features)
        elif isinstance(self.max_features, int):
            if self.random_state:
                np.random.seed(self.random_state)
            features_to_consider = np.random.choice(n_features, 
                                                   min(self.max_features, n_features), 
                                                   replace=False)
        elif self.max_features == 'sqrt':
            n_features_to_consider = int(np.sqrt(n_features))
            if self.random_state:
                np.random.seed(self.random_state)
            features_to_consider = np.random.choice(n_features, n_features_to_consider, replace=False)
        elif self.max_features == 'log2':
            n_features_to_consider = int(np.log2(n_features))
            if self.random_state:
                np.random.seed(self.random_state)
            features_to_consider = np.random.choice(n_features, n_features_to_consider, replace=False)
        else:
            features_to_consider = range(n_features)
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        current_impurity = self._calculate_impurity(y)
        
        for feature_idx in features_to_consider:
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try each possible threshold
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                
                # Calculate information gain
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """
        Recursively build the decision tree.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets
        depth : int
            Current depth of the tree
            
        Returns:
        --------
        node : DecisionTreeNode
            Root node of the subtree
        """
        n_samples, n_features = X.shape
        
        # Create leaf node if stopping criteria are met
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (len(np.unique(y)) == 1):
            
            if self.task == 'classification':
                # Most frequent class and class counts
                classes, counts = np.unique(y, return_counts=True)
                value = classes[np.argmax(counts)]
                class_counts = dict(zip(classes, counts))
            else:
                # Mean for regression
                value = np.mean(y)
                class_counts = None
            
            return DecisionTreeNode(value=value, samples=n_samples, class_counts=class_counts)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._get_best_split(X, y)
        
        # Create leaf node if no good split found
        if best_feature is None or best_gain <= 0:
            if self.task == 'classification':
                classes, counts = np.unique(y, return_counts=True)
                value = classes[np.argmax(counts)]
                class_counts = dict(zip(classes, counts))
            else:
                value = np.mean(y)
                class_counts = None
            
            return DecisionTreeNode(value=value, samples=n_samples, class_counts=class_counts)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(feature_index=best_feature, threshold=best_threshold,
                               left=left_child, right=right_child, samples=n_samples)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Fit the decision tree to the training data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training targets
            
        Returns:
        --------
        self : DecisionTree
            Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        # Build the tree
        self.tree = self._build_tree(X, y)
        
        return self
    
    def _predict_sample(self, x: np.ndarray) -> Union[int, float]:
        """
        Predict a single sample.
        
        Parameters:
        -----------
        x : np.ndarray
            Single sample to predict
            
        Returns:
        --------
        prediction : Union[int, float]
            Predicted value
        """
        node = self.tree
        
        if node is None:
            raise ValueError("Decision tree not fitted yet")
        
        while node.left is not None:
            if node.feature_index is None or node.threshold is None:
                break
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        if node.value is None:
            raise ValueError("Invalid tree structure")
        
        return node.value
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for the input data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        """
        if self.tree is None:
            raise ValueError("Tree hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        predictions = np.array([self._predict_sample(x) for x in X])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if self.tree is None:
            raise ValueError("Decision tree not fitted yet")
        
        X = np.array(X)
        probabilities = []
        
        for sample in X:
            node = self.tree
            
            # Traverse tree to find leaf
            while node is not None and (node.left is not None or node.right is not None):
                if node.feature_index is None or node.threshold is None:
                    break
                if sample[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            
            if node is None:
                # Fallback: uniform distribution
                proba = np.ones(self.n_classes_) / self.n_classes_
                probabilities.append(proba)
                continue
            
            # Get class probabilities from leaf node
            if node.class_counts is not None:
                total_samples = sum(node.class_counts.values())
                proba = np.zeros(self.n_classes_)
                if self.classes_ is not None:
                    for class_idx, class_label in enumerate(self.classes_):
                        count = node.class_counts.get(class_label, 0)
                        proba[class_idx] = count / total_samples if total_samples > 0 else 0
                probabilities.append(proba)
            else:
                # Fallback: create one-hot encoding
                predicted_class = node.value
                proba = np.zeros(self.n_classes_)
                if self.classes_ is not None and predicted_class is not None:
                    class_indices = np.where(self.classes_ == predicted_class)[0]
                    if len(class_indices) > 0:
                        proba[class_indices[0]] = 1.0
                probabilities.append(proba)
        
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
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def get_depth(self) -> int:
        """Get the depth of the tree."""
        if self.tree is None:
            return 0
        
        def _get_depth(node: DecisionTreeNode) -> int:
            if node.left is None and node.right is None:
                return 1
            
            left_depth = _get_depth(node.left) if node.left else 0
            right_depth = _get_depth(node.right) if node.right else 0
            
            return 1 + max(left_depth, right_depth)
        
        return _get_depth(self.tree)
    
    def get_n_leaves(self) -> int:
        """Get the number of leaf nodes."""
        if self.tree is None:
            return 0
        
        def _count_leaves(node: DecisionTreeNode) -> int:
            if node.left is None and node.right is None:
                return 1
            
            left_leaves = _count_leaves(node.left) if node.left else 0
            right_leaves = _count_leaves(node.right) if node.right else 0
            
            return left_leaves + right_leaves
        
        return _count_leaves(self.tree)
    
    def feature_importances_(self) -> np.ndarray:
        """
        Calculate feature importances based on impurity decrease.
        
        Returns:
        --------
        importances : np.ndarray
            Feature importances
        """
        if self.tree is None:
            raise ValueError("Tree hasn't been fitted yet. Call fit() first.")
        
        importances = np.zeros(self.n_features_)
        
        def _calculate_importances(node: DecisionTreeNode, total_samples: int):
            if node.left is None and node.right is None:
                return
            
            # Skip if feature_index is None (shouldn't happen in a well-formed tree)
            if node.feature_index is None:
                return
            
            # Calculate importance for this node
            left_samples = node.left.samples if node.left else 0
            right_samples = node.right.samples if node.right else 0
            
            # This is a simplified importance calculation
            # In practice, you'd calculate the actual impurity decrease
            importance = node.samples / total_samples
            importances[node.feature_index] += importance
            
            # Recursively calculate for children
            if node.left:
                _calculate_importances(node.left, total_samples)
            if node.right:
                _calculate_importances(node.right, total_samples)
        
        _calculate_importances(self.tree, self.tree.samples)
        
        # Normalize importances
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        return importances


def generate_tree_data(task: str = 'classification', n_samples: int = 200,
                      n_features: int = 2, noise: float = 0.1,
                      random_state: Optional[int] = None) -> tuple:
    """
    Generate synthetic data suitable for decision trees.
    
    Parameters:
    -----------
    task : str
        'classification' or 'regression'
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
    X, y : tuple
        Features and targets
    """
    if random_state:
        np.random.seed(random_state)
    
    X = np.random.uniform(-3, 3, (n_samples, n_features))
    
    if task == 'classification':
        # Create non-linear decision boundary
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if n_features >= 2:
                if X[i, 0] > 0 and X[i, 1] > 0:
                    y[i] = 1
                elif X[i, 0] < 0 and X[i, 1] < 0:
                    y[i] = 1
                else:
                    y[i] = 0
            else:
                y[i] = 1 if X[i, 0] > 0 else 0
        
        # Add noise
        noise_mask = np.random.random(n_samples) < noise
        y[noise_mask] = 1 - y[noise_mask]
        
        return X, y.astype(int)
    
    else:  # regression
        # Create non-linear target
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if n_features >= 2:
                y[i] = X[i, 0] ** 2 + X[i, 1] ** 2
            else:
                y[i] = X[i, 0] ** 2
        
        # Add noise
        y += np.random.normal(0, noise, n_samples)
        
        return X, y


class DecisionTreeVisualization:
    """
    Visualization utilities for Decision Trees.
    """
    
    @staticmethod
    def plot_tree_structure(tree: DecisionTree, feature_names: Optional[list] = None,
                           max_depth: int = 3) -> None:
        """
        Plot the tree structure (text-based).
        
        Parameters:
        -----------
        tree : DecisionTree
            Fitted decision tree
        feature_names : list, optional
            Names of features
        max_depth : int
            Maximum depth to display
        """
        if tree.tree is None:
            print("Tree hasn't been fitted yet.")
            return
        
        if feature_names is None:
            if tree.n_features_ is not None:
                feature_names = [f"feature_{i}" for i in range(tree.n_features_)]
            else:
                feature_names = []
        
        def _print_tree(node: DecisionTreeNode, depth: int = 0, prefix: str = ""):
            if depth > max_depth:
                return
            
            indent = "  " * depth
            
            if node.left is None and node.right is None:
                # Leaf node
                print(f"{indent}{prefix}Leaf: {node.value} (samples: {node.samples})")
            else:
                # Internal node
                if node.feature_index is not None and node.threshold is not None:
                    feature_name = feature_names[node.feature_index] if node.feature_index < len(feature_names) else f"feature_{node.feature_index}"
                    print(f"{indent}{prefix}{feature_name} <= {node.threshold:.2f} (samples: {node.samples})")
                else:
                    print(f"{indent}{prefix}Internal node (samples: {node.samples})")
                
                if node.left:
                    _print_tree(node.left, depth + 1, "L: ")
                if node.right:
                    _print_tree(node.right, depth + 1, "R: ")
        
        print("Decision Tree Structure:")
        print("=" * 30)
        _print_tree(tree.tree)
    
    @staticmethod
    def plot_decision_boundary(tree: DecisionTree, X: np.ndarray, y: np.ndarray,
                             title: str = "Decision Tree Decision Boundary") -> None:
        """
        Plot decision boundary for 2D data.
        
        Parameters:
        -----------
        tree : DecisionTree
            Fitted decision tree
        X : np.ndarray
            Training features (2D)
        y : np.ndarray
            Training targets
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
        Z = tree.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.6, cmap='RdYlBu')
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    
    @staticmethod
    def plot_feature_importances(tree: DecisionTree, feature_names: Optional[list] = None) -> None:
        """
        Plot feature importances.
        
        Parameters:
        -----------
        tree : DecisionTree
            Fitted decision tree
        feature_names : list, optional
            Names of features
        """
        if tree.tree is None:
            print("Tree hasn't been fitted yet.")
            return
        
        importances = tree.feature_importances_()
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title("Feature Importances")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()
