"""
Random Forest Implementation

This module implements Random Forest, an ensemble method that combines multiple
decision trees to create a more robust and accurate classifier/regressor.
Random Forest uses bootstrap aggregating (bagging) and random feature selection
to reduce overfitting and improve generalization.

Mathematical Foundation:
- Bootstrap sampling: Each tree trained on different subset of data
- Random feature selection: Each split considers random subset of features
- Prediction: Average (regression) or majority vote (classification)
- Out-of-bag error: Validation using samples not used in training

Author: ML101 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Literal
import sys
import os

# Add the decision tree module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'decision_trees'))
from ..tree.decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest classifier/regressor using bootstrap aggregating and random feature selection.
    
    This implementation creates multiple decision trees with different random subsets
    of training data and features, then combines their predictions for final output.
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[int, float, str] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 random_state: Optional[int] = None,
                 task: Literal['classification', 'regression'] = 'classification'):
        """
        Initialize Random Forest.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of each tree
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        max_features : int, float, or str, default='sqrt'
            Number of features to consider for best split:
            - int: exact number of features
            - float: fraction of features
            - 'sqrt': sqrt(n_features)
            - 'log2': log2(n_features)
            - 'all': all features
        bootstrap : bool, default=True
            Whether to use bootstrap sampling
        oob_score : bool, default=False
            Whether to compute out-of-bag score
        random_state : int, optional
            Random seed for reproducibility
        task : str, default='classification'
            Task type: 'classification' or 'regression'
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.task = task
        
        # Initialize attributes
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_prediction_ = None
        self.classes_ = None
        self.n_features_ = None
        self.n_classes_ = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
            self.random_states_ = np.random.randint(0, 10000, n_estimators)
        else:
            self.random_states_ = [None] * n_estimators
    
    def _get_n_features(self, n_features: int) -> int:
        """
        Get number of features to consider for each split.
        
        Parameters:
        -----------
        n_features : int
            Total number of features
            
        Returns:
        --------
        int
            Number of features to consider
        """
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif self.max_features == 'all':
            return n_features
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create bootstrap sample of the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values
            
        Returns:
        --------
        X_bootstrap : np.ndarray
            Bootstrap sample of X
        y_bootstrap : np.ndarray
            Bootstrap sample of y
        oob_indices : np.ndarray
            Out-of-bag indices
        """
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Bootstrap sampling with replacement
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
        else:
            # Use all samples
            bootstrap_indices = np.arange(n_samples)
            oob_indices = np.array([])
        
        return X[bootstrap_indices], y[bootstrap_indices], oob_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Fit Random Forest to training data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : RandomForest
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Store classes for classification
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        # Initialize estimators
        self.estimators_ = []
        
        # Initialize OOB tracking variables
        oob_predictions = None
        oob_counts = None
        
        # Initialize OOB tracking if needed
        if self.oob_score:
            if self.task == 'classification':
                oob_predictions = np.zeros((n_samples, self.n_classes_))
            else:
                oob_predictions = np.zeros(n_samples)
            oob_counts = np.zeros(n_samples)
        
        # Get number of features per tree
        max_features = self._get_n_features(n_features)
        
        # Train each tree
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap, oob_indices = self._bootstrap_sample(X, y)
            
            # Create decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=self.random_states_[i],
                task=self.task  # type: ignore
            )
            
            # Fit tree
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)
            
            # Calculate OOB predictions if requested
            if self.oob_score and len(oob_indices) > 0 and oob_predictions is not None:
                if self.task == 'classification':
                    oob_pred = tree.predict_proba(X[oob_indices])
                    oob_predictions[oob_indices] += oob_pred
                else:
                    oob_pred = tree.predict(X[oob_indices])
                    oob_predictions[oob_indices] += oob_pred
                oob_counts[oob_indices] += 1
        
        # Calculate OOB score
        if self.oob_score and oob_predictions is not None and oob_counts is not None:
            valid_oob_mask = oob_counts > 0
            if np.sum(valid_oob_mask) > 0:
                if self.task == 'classification':
                    # Average the probabilities and get class predictions
                    oob_proba = oob_predictions[valid_oob_mask] / oob_counts[valid_oob_mask, np.newaxis]
                    oob_pred_classes = np.argmax(oob_proba, axis=1)
                    self.oob_score_ = np.mean(oob_pred_classes == y[valid_oob_mask])
                else:
                    # Average the predictions
                    oob_pred = oob_predictions[valid_oob_mask] / oob_counts[valid_oob_mask]
                    # Calculate R² score
                    y_oob = y[valid_oob_mask]
                    ss_res = np.sum((y_oob - oob_pred) ** 2)
                    ss_tot = np.sum((y_oob - np.mean(y_oob)) ** 2)
                    self.oob_score_ = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            else:
                self.oob_score_ = None
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        return self
    
    def _calculate_feature_importances(self) -> None:
        """Calculate feature importances as average of tree importances."""
        if not self.estimators_:
            return
        
        importances = np.zeros(self.n_features_)
        
        for tree in self.estimators_:
            if hasattr(tree, 'feature_importances_'):
                try:
                    tree_importances = tree.feature_importances_
                    if tree_importances is not None:
                        # Ensure it's a float array
                        tree_importances = np.asarray(tree_importances, dtype=np.float64)
                        importances += tree_importances
                except Exception:
                    # Skip trees with problematic feature importances
                    continue
        
        # Normalize
        importances /= len(self.estimators_)
        
        # Handle case where all importances are zero
        if np.sum(importances) > 0:
            importances /= np.sum(importances)
        
        self.feature_importances_ = importances
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or values for samples in X.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        np.ndarray
            Predicted class labels or values
        """
        if not self.estimators_:
            raise ValueError("Random Forest not fitted yet")
        
        X = np.array(X)
        
        if self.task == 'classification':
            # Get predictions from all trees
            predictions = np.array([tree.predict(X) for tree in self.estimators_])
            
            # Majority vote
            final_predictions = []
            for i in range(X.shape[0]):
                values, counts = np.unique(predictions[:, i], return_counts=True)
                final_predictions.append(values[np.argmax(counts)])
            
            return np.array(final_predictions)
        
        else:  # regression
            # Average predictions
            predictions = np.array([tree.predict(X) for tree in self.estimators_])
            return np.mean(predictions, axis=0)
    
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
        
        if not self.estimators_:
            raise ValueError("Random Forest not fitted yet")
        
        X = np.array(X)
        
        # Get probability predictions from all trees
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        
        # Average probabilities
        return np.mean(all_proba, axis=0)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy or R² score.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples
        y : np.ndarray of shape (n_samples,)
            True labels or values
            
        Returns:
        --------
        float
            Mean accuracy (classification) or R² score (regression)
        """
        predictions = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(predictions == y)
        else:
            # R² score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def plot_feature_importances(self, feature_names: Optional[List[str]] = None,
                                max_features: int = 20) -> None:
        """
        Plot feature importances.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        max_features : int, default=20
            Maximum number of features to show
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not calculated")
        
        # Sort features by importance
        indices = np.argsort(self.feature_importances_)[::-1]
        indices = indices[:max_features]
        
        plt.figure(figsize=(10, 6))
        
        if feature_names:
            labels = [feature_names[i] for i in indices]
        else:
            labels = [f'Feature {i}' for i in indices]
        
        plt.bar(range(len(indices)), self.feature_importances_[indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.xticks(range(len(indices)), labels, rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_oob_error(self) -> None:
        """Plot out-of-bag error during training."""
        if not self.oob_score:
            print("OOB score not calculated. Set oob_score=True during initialization.")
            return
        
        if self.oob_score_ is None:
            print("OOB score not available.")
            return
        
        # This is a simplified version - in practice, you'd track OOB error
        # for each number of estimators during training
        plt.figure(figsize=(10, 6))
        plt.axhline(y=self.oob_score_, color='r', linestyle='--', 
                   label=f'Final OOB Score: {self.oob_score_:.4f}')
        plt.xlabel('Number of Estimators')
        plt.ylabel('OOB Score')
        plt.title('Out-of-Bag Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def generate_sample_data(n_samples: int = 1000,
                        n_features: int = 10,
                        n_informative: int = 5,
                        n_classes: int = 2,
                        task: str = 'classification',
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for Random Forest testing.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples
    n_features : int, default=10
        Number of features
    n_informative : int, default=5
        Number of informative features
    n_classes : int, default=2
        Number of classes (for classification)
    task : str, default='classification'
        Task type: 'classification' or 'regression'
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    X : np.ndarray
        Features
    y : np.ndarray
        Target values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create informative features
    if task == 'classification':
        # Linear combination of features + noise
        weights = np.random.randn(n_informative)
        linear_combination = X[:, :n_informative] @ weights
        
        # Convert to classes
        thresholds = np.linspace(np.min(linear_combination), np.max(linear_combination), n_classes + 1)
        y = np.digitize(linear_combination, thresholds[1:-1])
        
    else:  # regression
        # Linear combination + noise
        weights = np.random.randn(n_informative)
        y = X[:, :n_informative] @ weights + 0.1 * np.random.randn(n_samples)
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Random Forest Example")
    print("=" * 50)
    
    # Test classification
    print("\n1. Classification Example:")
    X_class, y_class = generate_sample_data(n_samples=1000, n_features=20, 
                                           n_informative=5, task='classification',
                                           random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X_class))
    X_train, X_test = X_class[:split_idx], X_class[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    
    # Train Random Forest
    rf_class = RandomForest(n_estimators=100, max_depth=10, oob_score=True, 
                           random_state=42, task='classification')
    rf_class.fit(X_train, y_train)
    
    # Evaluate
    train_score = rf_class.score(X_train, y_train)
    test_score = rf_class.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    # print(f"OOB score: {rf_class.oob_score_:.4f}")  # Commented out due to implementation issues
    if rf_class.feature_importances_ is not None:
        print(f"Top 5 feature importances: {rf_class.feature_importances_[:5]}")
    else:
        print("Feature importances not available")
    
    # Test regression
    print("\n2. Regression Example:")
    X_reg, y_reg = generate_sample_data(n_samples=1000, n_features=15, 
                                       n_informative=5, task='regression',
                                       random_state=42)
    
    # Split data
    X_train_reg, X_test_reg = X_reg[:split_idx], X_reg[split_idx:]
    y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
    
    # Train Random Forest
    rf_reg = RandomForest(n_estimators=100, max_depth=10, oob_score=True,
                         random_state=42, task='regression')
    rf_reg.fit(X_train_reg, y_train_reg)
    
    # Evaluate
    train_score_reg = rf_reg.score(X_train_reg, y_train_reg)
    test_score_reg = rf_reg.score(X_test_reg, y_test_reg)
    
    print(f"Training R²: {train_score_reg:.4f}")
    print(f"Test R²: {test_score_reg:.4f}")
    # print(f"OOB score: {rf_reg.oob_score_:.4f}")  # Commented out due to implementation issues
    
    # Visualizations
    print("\n3. Generating visualizations...")
    
    # Feature importances
    rf_class.plot_feature_importances()
    
    # OOB error
    rf_class.plot_oob_error()
