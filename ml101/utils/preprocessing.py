"""
Data Preprocessing Utilities

This module implements common data preprocessing techniques from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
    z = (x - μ) / σ
    
    where μ is the mean of the training samples and σ is the standard deviation.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            
        Returns:
        --------
        self : StandardScaler
            Returns self for method chaining
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        self.n_features_ = X.shape[1]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to transform
            
        Returns:
        --------
        X_scaled : np.ndarray
            Transformed data
        """
        if self.mean_ is None:
            raise ValueError("Scaler hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
            
        Returns:
        --------
        X_scaled : np.ndarray
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation.
        
        Parameters:
        -----------
        X : np.ndarray
            Scaled data
            
        Returns:
        --------
        X_original : np.ndarray
            Data in original scale
        """
        if self.mean_ is None:
            raise ValueError("Scaler hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.
    
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between zero and one.
    
    The transformation is given by:
    X_scaled = (X - X.min) / (X.max - X.min) * (max - min) + min
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            
        Returns:
        --------
        self : MinMaxScaler
            Returns self for method chaining
        """
        X = np.array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Avoid division by zero
        self.data_range_ = np.where(self.data_range_ == 0, 1, self.data_range_)
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features according to feature_range.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data to transform
            
        Returns:
        --------
        X_scaled : np.ndarray
            Transformed data
        """
        if self.scale_ is None:
            raise ValueError("Scaler hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        return X * self.scale_ + self.min_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo the scaling of X according to feature_range.
        
        Parameters:
        -----------
        X : np.ndarray
            Scaled data
            
        Returns:
        --------
        X_original : np.ndarray
            Data in original scale
        """
        if self.scale_ is None:
            raise ValueError("Scaler hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        return (X - self.min_) / self.scale_


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    """
    
    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None
    
    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """
        Fit label encoder.
        
        Parameters:
        -----------
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : LabelEncoder
            Returns self for method chaining
        """
        self.classes_ = np.unique(y)
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels to normalized encoding.
        
        Parameters:
        -----------
        y : np.ndarray
            Target values
            
        Returns:
        --------
        y_encoded : np.ndarray
            Encoded labels
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder hasn't been fitted yet. Call fit() first.")
        
        return np.array([self.class_to_index_[label] for label in y])
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit label encoder and return encoded labels."""
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels back to original encoding.
        
        Parameters:
        -----------
        y : np.ndarray
            Encoded labels
            
        Returns:
        --------
        y_original : np.ndarray
            Original labels
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder hasn't been fitted yet. Call fit() first.")
        
        return np.array([self.classes_[idx] for idx in y])


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    """
    
    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first
        self.categories_ = None
        self.n_features_ = None
    
    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """
        Fit OneHotEncoder to X.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to determine the categories of each feature
            
        Returns:
        --------
        self : OneHotEncoder
            Returns self for method chaining
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_ = X.shape[1]
        self.categories_ = []
        
        for i in range(self.n_features_):
            categories = np.unique(X[:, i])
            if self.drop_first and len(categories) > 1:
                categories = categories[1:]  # Drop first category
            self.categories_.append(categories)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X using one-hot encoding.
        
        Parameters:
        -----------
        X : np.ndarray
            The data to encode
            
        Returns:
        --------
        X_encoded : np.ndarray
            Transformed input
        """
        if self.categories_ is None:
            raise ValueError("OneHotEncoder hasn't been fitted yet. Call fit() first.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        encoded_features = []
        
        for i in range(self.n_features_):
            feature_categories = self.categories_[i]
            feature_encoded = np.zeros((X.shape[0], len(feature_categories)))
            
            for j, category in enumerate(feature_categories):
                mask = X[:, i] == category
                feature_encoded[mask, j] = 1
            
            encoded_features.append(feature_encoded)
        
        return np.hstack(encoded_features)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit OneHotEncoder to X, then transform X."""
        return self.fit(X).transform(X)


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.
    
    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    """
    
    def __init__(self, degree: int = 2, include_bias: bool = True):
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None
    
    def fit(self, X: np.ndarray) -> 'PolynomialFeatures':
        """
        Compute number of output features.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The data
            
        Returns:
        --------
        self : PolynomialFeatures
            Returns self for method chaining
        """
        X = np.array(X)
        self.n_input_features_ = X.shape[1]
        
        # Calculate number of output features
        n_output_features = 0
        if self.include_bias:
            n_output_features += 1
        
        # Add original features
        n_output_features += self.n_input_features_
        
        # Add polynomial features
        for d in range(2, self.degree + 1):
            n_output_features += self._n_combinations_with_replacement(
                self.n_input_features_, d)
        
        self.n_output_features_ = n_output_features
        return self
    
    def _n_combinations_with_replacement(self, n: int, r: int) -> int:
        """Calculate combinations with replacement."""
        if r == 0:
            return 1
        return int(np.math.factorial(n + r - 1) / 
                  (np.math.factorial(r) * np.math.factorial(n - 1)))
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to polynomial features.
        
        Parameters:
        -----------
        X : np.ndarray
            The data to transform
            
        Returns:
        --------
        X_poly : np.ndarray
            The matrix of features
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        features = []
        
        # Add bias term
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        
        # Add original features
        features.append(X)
        
        # Add polynomial features
        for d in range(2, self.degree + 1):
            poly_features = self._generate_polynomial_features(X, d)
            features.append(poly_features)
        
        return np.hstack(features)
    
    def _generate_polynomial_features(self, X: np.ndarray, degree: int) -> np.ndarray:
        """Generate polynomial features for a specific degree."""
        n_samples, n_features = X.shape
        combinations = self._get_combinations_with_replacement(n_features, degree)
        
        poly_features = np.ones((n_samples, len(combinations)))
        
        for i, combination in enumerate(combinations):
            for feature_idx in combination:
                poly_features[:, i] *= X[:, feature_idx]
        
        return poly_features
    
    def _get_combinations_with_replacement(self, n: int, r: int) -> list:
        """Get all combinations with replacement."""
        if r == 0:
            return [()]
        
        combinations = []
        for i in range(n):
            for rest in self._get_combinations_with_replacement(n - i, r - 1):
                combinations.append((i,) + tuple(j + i for j in rest))
        
        return combinations
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PolynomialFeatures to X, then transform X."""
        return self.fit(X).transform(X)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """
    Split arrays into random train and test subsets.
    
    Parameters:
    -----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target variable
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Split data
    """
    if random_state:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Random permutation
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class PreprocessingVisualization:
    """
    Visualization utilities for preprocessing.
    """
    
    @staticmethod
    def plot_scaling_comparison(X: np.ndarray, feature_names: Optional[list] = None) -> None:
        """
        Compare different scaling methods.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        feature_names : list, optional
            Names of features
        """
        scalers = {
            'Original': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (name, scaler) in enumerate(scalers.items()):
            if scaler is None:
                X_scaled = X
            else:
                X_scaled = scaler.fit_transform(X)
            
            axes[i].boxplot(X_scaled, labels=feature_names)
            axes[i].set_title(f'{name}')
            axes[i].set_ylabel('Values')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Scaling Methods Comparison')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_polynomial_features(X: np.ndarray, y: np.ndarray, degrees: list = [1, 2, 3]) -> None:
        """
        Visualize polynomial feature transformation effects.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (should be 1D for visualization)
        y : np.ndarray
            Target values
        degrees : list
            Polynomial degrees to compare
        """
        if X.shape[1] != 1:
            raise ValueError("Can only visualize 1D input features")
        
        fig, axes = plt.subplots(1, len(degrees), figsize=(5 * len(degrees), 5))
        if len(degrees) == 1:
            axes = [axes]
        
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        
        for i, degree in enumerate(degrees):
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            X_plot_poly = poly.transform(X_plot)
            
            # Fit simple linear regression on polynomial features
            # Using normal equation: theta = (X^T X)^(-1) X^T y
            theta = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
            y_plot = X_plot_poly @ theta
            
            axes[i].scatter(X, y, alpha=0.6, label='Data')
            axes[i].plot(X_plot, y_plot, color='red', linewidth=2, 
                        label=f'Polynomial degree {degree}')
            axes[i].set_title(f'Polynomial Features (degree={degree})')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('y')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def preprocessing_example():
    """Example of using preprocessing utilities."""
    print("PREPROCESSING UTILITIES EXAMPLE")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 3) * [10, 5, 2] + [50, 20, 5]
    y = np.random.choice(['A', 'B', 'C'], 100)
    
    print("\n--- SCALING EXAMPLES ---")
    print("Original data statistics:")
    print(f"Mean: {np.mean(X, axis=0)}")
    print(f"Std:  {np.std(X, axis=0)}")
    print(f"Min:  {np.min(X, axis=0)}")
    print(f"Max:  {np.max(X, axis=0)}")
    
    # Standard scaling
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X)
    print(f"\nAfter StandardScaler:")
    print(f"Mean: {np.mean(X_std, axis=0)}")
    print(f"Std:  {np.std(X_std, axis=0)}")
    
    # MinMax scaling
    scaler_minmax = MinMaxScaler()
    X_minmax = scaler_minmax.fit_transform(X)
    print(f"\nAfter MinMaxScaler:")
    print(f"Min: {np.min(X_minmax, axis=0)}")
    print(f"Max: {np.max(X_minmax, axis=0)}")
    
    print("\n--- ENCODING EXAMPLES ---")
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Original labels: {y[:10]}")
    print(f"Encoded labels:  {y_encoded[:10]}")
    print(f"Classes: {label_encoder.classes_}")
    
    # One-hot encoding
    onehot_encoder = OneHotEncoder()
    y_onehot = onehot_encoder.fit_transform(y.reshape(-1, 1))
    print(f"\nOne-hot encoded shape: {y_onehot.shape}")
    print(f"First 5 samples:\n{y_onehot[:5]}")
    
    print("\n--- POLYNOMIAL FEATURES ---")
    # Polynomial features (1D example)
    X_1d = np.random.uniform(-2, 2, 50).reshape(-1, 1)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_1d)
    print(f"Original features shape: {X_1d.shape}")
    print(f"Polynomial features shape: {X_poly.shape}")
    
    print("\n--- TRAIN/TEST SPLIT ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")


if __name__ == "__main__":
    preprocessing_example()
