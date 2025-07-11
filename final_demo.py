#!/usr/bin/env python3
"""
ML101 Package - Final Demo and Usage Examples

This script demonstrates the complete functionality of the ML101 package
and shows how users will interact with it after installation from PyPI.
"""

import numpy as np
import matplotlib.pyplot as plt
from ml101 import (
    LinearRegression, LogisticRegression, RidgeRegression, LassoRegression,
    KNearestNeighbors, DecisionTree, RandomForest, KMeans, 
    GaussianNaiveBayes, PCA
)
from ml101.utils import StandardScaler, MinMaxScaler, train_test_split

def header(title):
    """Print a nice header for each section."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def demo_linear_models():
    """Demonstrate linear models."""
    header("LINEAR MODELS DEMONSTRATION")
    
    # Generate regression data
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = 2*X[:, 0] + 3*X[:, 1] + 0.5*X[:, 2] + np.random.randn(200) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    print(f"Linear Regression R² Score: {lr_score:.4f}")
    
    # Ridge Regression
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_score = ridge.score(X_test, y_test)
    print(f"Ridge Regression R² Score: {ridge_score:.4f}")
    
    # Lasso Regression
    lasso = LassoRegression(alpha=0.1)
    lasso.fit(X_train, y_train)
    lasso_score = lasso.score(X_test, y_test)
    print(f"Lasso Regression R² Score: {lasso_score:.4f}")
    
    # Classification data
    y_clf = (y > np.median(y)).astype(int)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.3, random_state=42)
    
    # Logistic Regression
    logistic = LogisticRegression()
    logistic.fit(X_train, y_clf_train)
    logistic_score = logistic.score(X_test, y_clf_test)
    print(f"Logistic Regression Accuracy: {logistic_score:.4f}")

def demo_tree_models():
    """Demonstrate tree-based models."""
    header("TREE-BASED MODELS DEMONSTRATION")
    
    # Generate classification data
    np.random.seed(42)
    X = np.random.randn(300, 4)
    y = ((X[:, 0] + X[:, 1]) > (X[:, 2] + X[:, 3])).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Decision Tree
    dt = DecisionTree(task='classification', max_depth=5)
    dt.fit(X_train, y_train)
    dt_score = dt.score(X_test, y_test)
    print(f"Decision Tree Accuracy: {dt_score:.4f}")
    
    # Random Forest
    rf = RandomForest(n_estimators=20, max_depth=5, task='classification', random_state=42)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_score:.4f}")

def demo_neighbors_and_clustering():
    """Demonstrate KNN and clustering."""
    header("NEIGHBORS & CLUSTERING DEMONSTRATION")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # K-Nearest Neighbors
    knn = KNearestNeighbors(k=5, task='classification')
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print(f"KNN Accuracy: {knn_score:.4f}")
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    print(f"K-Means found {len(np.unique(labels))} clusters")
    print(f"Inertia: {kmeans.inertia_:.4f}")

def demo_naive_bayes():
    """Demonstrate Naive Bayes."""
    header("NAIVE BAYES DEMONSTRATION")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    gnb_score = gnb.score(X_test, y_test)
    probabilities = gnb.predict_proba(X_test)
    print(f"Gaussian Naive Bayes Accuracy: {gnb_score:.4f}")
    print(f"First 5 prediction probabilities:\n{probabilities[:5]}")

def demo_dimensionality_reduction():
    """Demonstrate PCA."""
    header("DIMENSIONALITY REDUCTION DEMONSTRATION")
    
    # Generate high-dimensional data
    np.random.seed(42)
    X = np.random.randn(150, 10)
    
    # PCA
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {X_reduced.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

def demo_preprocessing():
    """Demonstrate preprocessing utilities."""
    header("PREPROCESSING UTILITIES DEMONSTRATION")
    
    # Generate data with different scales
    np.random.seed(42)
    X = np.random.randn(100, 3) * [1, 10, 100] + [0, 50, 1000]
    
    print("Original data statistics:")
    print(f"Mean: {np.mean(X, axis=0)}")
    print(f"Std:  {np.std(X, axis=0)}")
    
    # StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    print(f"\nAfter StandardScaler:")
    print(f"Mean: {np.mean(X_std, axis=0)}")
    print(f"Std:  {np.std(X_std, axis=0)}")
    
    # MinMaxScaler
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    print(f"\nAfter MinMaxScaler:")
    print(f"Min: {np.min(X_minmax, axis=0)}")
    print(f"Max: {np.max(X_minmax, axis=0)}")

def main():
    """Run all demonstrations."""
    print("🎯 ML101 PACKAGE - COMPREHENSIVE DEMONSTRATION")
    print("📦 Package Version: 0.1.0")
    print("🚀 Ready for PyPI Publishing!")
    
    # Run all demos
    demo_linear_models()
    demo_tree_models()
    demo_neighbors_and_clustering()
    demo_naive_bayes()
    demo_dimensionality_reduction()
    demo_preprocessing()
    
    header("PACKAGE SUMMARY")
    print("✅ 9 Core ML Algorithms Implemented:")
    print("   • Linear Models: LinearRegression, LogisticRegression, RidgeRegression, LassoRegression")
    print("   • Tree Models: DecisionTree, RandomForest")
    print("   • Neighbors: KNearestNeighbors")
    print("   • Clustering: KMeans")
    print("   • Naive Bayes: GaussianNaiveBayes")
    print("   • Dimensionality Reduction: PCA")
    print("   • Utilities: StandardScaler, MinMaxScaler, train_test_split")
    
    print("\n📖 Features:")
    print("   • Pure NumPy implementations for educational transparency")
    print("   • Scikit-learn compatible API")
    print("   • Comprehensive documentation")
    print("   • Mathematical foundations explained")
    print("   • Ready for pip installation")
    
    print("\n🎉 Package is ready for PyPI publication!")
    print("\nInstallation command after publishing:")
    print("   pip install ml101-algorithms")
    
    print("\nUsage example:")
    print("   from ml101 import LinearRegression")
    print("   from ml101.utils import train_test_split")
    print("   model = LinearRegression()")
    print("   model.fit(X_train, y_train)")

if __name__ == "__main__":
    main()
