"""
ML101: Classical Machine Learning Algorithms from Scratch

A comprehensive, educational library of machine learning algorithms implemented 
from scratch. Perfect for learning, teaching, and understanding the inner workings 
of classical ML algorithms.

Features:
- Clean, readable implementations
- Comprehensive documentation
- Educational focus with mathematical explanations
- Compatible with scikit-learn API
- Extensive examples and tutorials

Author: ML101 Contributors
License: MIT
"""

__version__ = "0.1.0"
__author__ = "ML101 Contributors"
__email__ = "ml101.algorithms@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/hustcalm/ML101"

# Import main classes for easy access
from .linear_models import LinearRegression, LogisticRegression, RidgeRegression, LassoRegression
from .neighbors import KNearestNeighbors
from .tree import DecisionTree
from .ensemble import RandomForest
from .clustering import KMeans
from .naive_bayes import GaussianNaiveBayes, MultinomialNaiveBayes, BernoulliNaiveBayes
from .decomposition import PCA
from .svm import SVM
from .utils import StandardScaler, MinMaxScaler, train_test_split

__all__ = [
    # Linear Models
    'LinearRegression',
    'LogisticRegression', 
    'RidgeRegression',
    'LassoRegression',
    
    # Neighbors
    'KNearestNeighbors',
    
    # Trees and Ensembles
    'DecisionTree',
    'RandomForest',
    
    # Clustering
    'KMeans',
    
    # Naive Bayes
    'GaussianNaiveBayes',
    'MultinomialNaiveBayes',
    'BernoulliNaiveBayes',
    
    # Decomposition
    'PCA',
    
    # Support Vector Machines
    'SVM',
    
    # Utilities
    'StandardScaler',
    'MinMaxScaler',
    'train_test_split',
]

# Version information
def get_version():
    """Get the current version of ML101."""
    return __version__

# Package information
def get_info():
    """Get package information."""
    return {
        'name': 'ML101',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'url': __url__,
        'description': 'Educational machine learning algorithms from scratch'
    }
