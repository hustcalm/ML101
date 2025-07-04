"""
Regularized Regression Comparison

This script demonstrates and compares different regularized regression techniques:
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Linear Regression (no regularization) for comparison

Author: ML101 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add algorithm paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms', 'linear_regression'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms', 'ridge_regression'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms', 'lasso_regression'))

from linear_regression import LinearRegression
from ridge_regression import RidgeRegression
from lasso_regression import LassoRegression


def compare_regularization_methods():
    """Compare Ridge, Lasso, and Linear regression on synthetic data."""
    print("=" * 60)
    print("REGULARIZED REGRESSION COMPARISON")
    print("=" * 60)
    
    # Generate data with multicollinearity and sparse ground truth
    np.random.seed(42)
    n_samples, n_features = 100, 15
    n_informative = 5
    
    # Create feature matrix with correlations
    X = np.random.randn(n_samples, n_features)
    
    # Add multicollinearity
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] - 0.3 * X[:, 1] + 0.3 * np.random.randn(n_samples)
    
    # Create sparse coefficient vector
    true_coef = np.zeros(n_features)
    informative_indices = np.random.choice(n_features, n_informative, replace=False)
    true_coef[informative_indices] = np.random.randn(n_informative) * 2
    
    # Generate target
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"True informative features: {n_informative}")
    
    return X_train, X_test, y_train, y_test, true_coef


def demonstrate_ridge_regression(X_train, X_test, y_train, y_test):
    """Demonstrate Ridge regression with different alpha values."""
    print("\n" + "=" * 50)
    print("RIDGE REGRESSION (L2 REGULARIZATION)")
    print("=" * 50)
    
    alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
    
    print(f"{'Alpha':>8} {'Train R²':>10} {'Test R²':>9} {'Coef Norm':>10}")
    print("-" * 40)
    
    best_alpha = None
    best_test_score = -np.inf
    
    for alpha in alphas:
        ridge = RidgeRegression(alpha=alpha, solver='normal')
        ridge.fit(X_train, y_train)
        
        train_score = ridge.score(X_train, y_train)
        test_score = ridge.score(X_test, y_test)
        coef_norm = np.linalg.norm(ridge.coef_)
        
        print(f"{alpha:8.1f} {train_score:10.4f} {test_score:9.4f} {coef_norm:10.4f}")
        
        if test_score > best_test_score:
            best_test_score = test_score
            best_alpha = alpha
    
    print(f"\nBest Ridge alpha: {best_alpha} (Test R² = {best_test_score:.4f})")
    
    return best_alpha


def demonstrate_lasso_regression(X_train, X_test, y_train, y_test, true_coef):
    """Demonstrate Lasso regression with different alpha values."""
    print("\n" + "=" * 50)
    print("LASSO REGRESSION (L1 REGULARIZATION)")
    print("=" * 50)
    
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    print(f"{'Alpha':>8} {'Train R²':>10} {'Test R²':>9} {'Features':>9} {'Sparsity':>9}")
    print("-" * 50)
    
    best_alpha = None
    best_test_score = -np.inf
    
    for alpha in alphas:
        lasso = LassoRegression(alpha=alpha, max_iter=1000)
        lasso.fit(X_train, y_train)
        
        train_score = lasso.score(X_train, y_train)
        test_score = lasso.score(X_test, y_test)
        n_features = len(lasso.get_selected_features())
        sparsity = np.sum(np.abs(lasso.coef_) < 1e-6) / len(lasso.coef_)
        
        print(f"{alpha:8.3f} {train_score:10.4f} {test_score:9.4f} {n_features:9d} {sparsity:8.2%}")
        
        if test_score > best_test_score:
            best_test_score = test_score
            best_alpha = alpha
    
    print(f"\nBest Lasso alpha: {best_alpha} (Test R² = {best_test_score:.4f})")
    
    # Feature selection analysis
    lasso_best = LassoRegression(alpha=best_alpha, max_iter=1000)
    lasso_best.fit(X_train, y_train)
    
    selected_features = lasso_best.get_selected_features()
    true_informative = np.where(np.abs(true_coef) > 1e-6)[0]
    
    print(f"\nFeature Selection Analysis:")
    print(f"True informative features: {sorted(true_informative)}")
    print(f"Lasso selected features: {sorted(selected_features)}")
    
    # Calculate precision and recall
    tp = len(set(selected_features) & set(true_informative))
    fp = len(set(selected_features) - set(true_informative))
    fn = len(set(true_informative) - set(selected_features))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    return best_alpha


def compare_methods(X_train, X_test, y_train, y_test, ridge_alpha, lasso_alpha):
    """Final comparison of all methods with best parameters."""
    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)
    
    # Linear Regression
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    
    # Ridge Regression
    ridge = RidgeRegression(alpha=ridge_alpha, solver='normal')
    ridge.fit(X_train, y_train)
    
    # Lasso Regression
    lasso = LassoRegression(alpha=lasso_alpha, max_iter=1000)
    lasso.fit(X_train, y_train)
    
    models = [
        ("Linear Regression", linear),
        ("Ridge Regression", ridge),
        ("Lasso Regression", lasso)
    ]
    
    print(f"{'Method':>18} {'Train R²':>10} {'Test R²':>9} {'Features':>9} {'Coef Norm':>10}")
    print("-" * 60)
    
    for name, model in models:
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        coef_norm = np.linalg.norm(model.coef_)
        
        if hasattr(model, 'get_selected_features'):
            n_features = len(model.get_selected_features())
        else:
            n_features = len(model.coef_)
        
        print(f"{name:>18} {train_score:10.4f} {test_score:9.4f} {n_features:9d} {coef_norm:10.4f}")
    
    return models


def plot_coefficient_comparison(models, true_coef):
    """Plot coefficient comparison between methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # True coefficients
    axes[0, 0].bar(range(len(true_coef)), true_coef, alpha=0.7, color='green')
    axes[0, 0].set_title('True Coefficients')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Coefficient Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Model coefficients
    colors = ['blue', 'red', 'orange']
    for i, (name, model) in enumerate(models):
        row = (i + 1) // 2
        col = (i + 1) % 2
        
        axes[row, col].bar(range(len(model.coef_)), model.coef_, 
                          alpha=0.7, color=colors[i])
        axes[row, col].set_title(f'{name} Coefficients')
        axes[row, col].set_xlabel('Feature Index')
        axes[row, col].set_ylabel('Coefficient Value')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_regularization_paths(X_train, y_train):
    """Plot regularization paths for Ridge and Lasso."""
    alphas = np.logspace(-3, 2, 50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ridge regularization path
    ridge_coefs = []
    for alpha in alphas:
        ridge = RidgeRegression(alpha=alpha, solver='normal')
        ridge.fit(X_train, y_train)
        ridge_coefs.append(ridge.coef_)
    
    ridge_coefs = np.array(ridge_coefs)
    
    for i in range(ridge_coefs.shape[1]):
        ax1.plot(alphas, ridge_coefs[:, i], label=f'Feature {i}')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization Parameter (α)')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('Ridge Regression: Regularization Path')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Lasso regularization path
    lasso_coefs = []
    for alpha in alphas:
        lasso = LassoRegression(alpha=alpha, max_iter=1000)
        lasso.fit(X_train, y_train)
        lasso_coefs.append(lasso.coef_)
    
    lasso_coefs = np.array(lasso_coefs)
    
    for i in range(lasso_coefs.shape[1]):
        ax2.plot(alphas, lasso_coefs[:, i], label=f'Feature {i}')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Regularization Parameter (α)')
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title('Lasso Regression: Regularization Path')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration function."""
    # Generate data
    X_train, X_test, y_train, y_test, true_coef = compare_regularization_methods()
    
    # Demonstrate each method
    best_ridge_alpha = demonstrate_ridge_regression(X_train, X_test, y_train, y_test)
    best_lasso_alpha = demonstrate_lasso_regression(X_train, X_test, y_train, y_test, true_coef)
    
    # Final comparison
    models = compare_methods(X_train, X_test, y_train, y_test, 
                           best_ridge_alpha, best_lasso_alpha)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_coefficient_comparison(models, true_coef)
    plot_regularization_paths(X_train, y_train)
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("• Ridge Regression: Shrinks coefficients, handles multicollinearity")
    print("• Lasso Regression: Performs feature selection, creates sparse models")
    print("• Linear Regression: No regularization, can overfit with many features")
    print("• Choose method based on: interpretability needs, feature importance, data size")
    print("=" * 60)


if __name__ == "__main__":
    main()
