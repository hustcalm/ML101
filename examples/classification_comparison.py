"""
Comprehensive Machine Learning Examples

This script demonstrates multiple algorithms with comparisons and visualizations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogistic

# Import our implementations
from algorithms.linear_regression.linear_regression import LinearRegression, generate_linear_data
from algorithms.logistic_regression.logistic_regression import LogisticRegression, generate_classification_data
from algorithms.knn.knn import KNearestNeighbors, generate_knn_data, KNNVisualization


def linear_regression_demo():
    """Demonstrate Linear Regression with different methods."""
    print("=" * 60)
    print("LINEAR REGRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Generate data
    X, y = generate_linear_data(n_samples=150, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test both methods
    methods = ['normal', 'gradient_descent']
    models = {}
    
    for method in methods:
        print(f"\n--- {method.upper().replace('_', ' ')} METHOD ---")
        
        model = LinearRegression(method=method, learning_rate=0.01, max_iterations=1000)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Weight: {model.weights[0]:.4f}")
        print(f"Bias: {model.bias:.4f}")
        print(f"Training R²: {train_score:.4f}")
        print(f"Test R²: {test_score:.4f}")
        
        models[method] = model
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Data and predictions
    plt.subplot(1, 3, 1)
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    
    plt.scatter(X_train, y_train, alpha=0.6, label='Training Data', color='blue')
    plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', color='red')
    
    for method, color in zip(methods, ['green', 'orange']):
        y_plot = models[method].predict(X_plot)
        plt.plot(X_plot, y_plot, color=color, linewidth=2, 
                label=f'{method.replace("_", " ").title()}')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost history for gradient descent
    plt.subplot(1, 3, 2)
    if models['gradient_descent'].cost_history:
        plt.plot(models['gradient_descent'].cost_history, color='purple')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Gradient Descent Cost History')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    plt.subplot(1, 3, 3)
    y_pred = models['normal'].predict(X_test)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def logistic_regression_demo():
    """Demonstrate Logistic Regression."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Generate binary classification data
    X, y = generate_classification_data(n_samples=300, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Our implementation
    print("\n--- OUR IMPLEMENTATION ---")
    our_model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    our_model.fit(X_train_scaled, y_train)
    
    our_train_acc = our_model.score(X_train_scaled, y_train)
    our_test_acc = our_model.score(X_test_scaled, y_test)
    
    print(f"Training Accuracy: {our_train_acc:.4f}")
    print(f"Test Accuracy: {our_test_acc:.4f}")
    
    # Scikit-learn comparison
    print("\n--- SCIKIT-LEARN COMPARISON ---")
    sklearn_model = SklearnLogistic(random_state=42, max_iter=1000)
    sklearn_model.fit(X_train_scaled, y_train)
    
    sklearn_train_acc = sklearn_model.score(X_train_scaled, y_train)
    sklearn_test_acc = sklearn_model.score(X_test_scaled, y_test)
    
    print(f"Training Accuracy: {sklearn_train_acc:.4f}")
    print(f"Test Accuracy: {sklearn_test_acc:.4f}")
    
    # Visualize decision boundary
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Our implementation
    plt.subplot(1, 3, 1)
    plot_logistic_decision_boundary(our_model, X_train_scaled, y_train, scaler, "Our Implementation")
    
    # Plot 2: Cost history
    plt.subplot(1, 3, 2)
    if our_model.cost_history:
        plt.plot(our_model.cost_history, color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Logistic Loss')
        plt.title('Cost Function History')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Probability comparison
    plt.subplot(1, 3, 3)
    our_probs = our_model.predict_proba(X_test_scaled)[:, 1]
    sklearn_probs = sklearn_model.predict_proba(X_test_scaled)[:, 1]
    
    plt.scatter(our_probs, sklearn_probs, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
    plt.xlabel('Our Implementation Probabilities')
    plt.ylabel('Scikit-learn Probabilities')
    plt.title('Probability Predictions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_logistic_decision_boundary(model, X, y, scaler, title):
    """Plot decision boundary for logistic regression."""
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)[:, 1]
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')


def knn_demo():
    """Demonstrate K-Nearest Neighbors."""
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS DEMONSTRATION")
    print("=" * 60)
    
    # Classification demo
    print("\n--- CLASSIFICATION ---")
    X, y = generate_knn_data(task='classification', n_samples=300, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different k values
    k_values = [1, 3, 5, 10, 15]
    
    print("K-value analysis:")
    for k in k_values:
        model = KNearestNeighbors(k=k, task='classification')
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"k={k:2d}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
    
    # Visualize decision boundaries for different k values
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, k in enumerate([1, 3, 5, 10, 15]):
        model = KNearestNeighbors(k=k, task='classification')
        model.fit(X_train, y_train)
        
        ax = axes[i]
        plot_knn_decision_boundary(model, X_train, y_train, ax, f"KNN (k={k})")
    
    # K-value analysis plot
    KNNVisualization.plot_k_analysis(X_train, y_train, X_test, y_test, range(1, 21))
    
    # Regression demo
    print("\n--- REGRESSION ---")
    X_reg, y_reg = generate_knn_data(task='regression', n_samples=100, noise=0.1, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42)
    
    # Compare different k values for regression
    plt.figure(figsize=(15, 5))
    
    for i, k in enumerate([1, 5, 15]):
        model = KNearestNeighbors(k=k, task='regression')
        model.fit(X_reg_train, y_reg_train)
        
        # Create smooth prediction line
        X_plot = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        
        plt.subplot(1, 3, i+1)
        plt.scatter(X_reg_train, y_reg_train, alpha=0.6, label='Training Data', color='blue')
        plt.scatter(X_reg_test, y_reg_test, alpha=0.6, label='Test Data', color='red')
        plt.plot(X_plot, y_plot, color='green', linewidth=2, label=f'KNN (k={k})')
        
        test_score = model.score(X_reg_test, y_reg_test)
        plt.title(f'KNN Regression (k={k})\nR² = {test_score:.3f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_knn_decision_boundary(model, X, y, ax, title):
    """Plot KNN decision boundary."""
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def algorithm_comparison():
    """Compare different algorithms on the same dataset."""
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)
    
    # Generate classification data
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(learning_rate=0.1, max_iterations=1000),
        'KNN (k=5)': KNearestNeighbors(k=5, task='classification'),
        'KNN (k=1)': KNearestNeighbors(k=1, task='classification'),
        'KNN (k=15)': KNearestNeighbors(k=15, task='classification'),
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
        
        results[name] = {'train': train_acc, 'test': test_acc}
        print(f"{name:20s}: Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, model) in enumerate(models.items()):
        ax = axes[i]
        
        if 'Logistic' in name:
            plot_logistic_boundary_on_ax(model, X_train_scaled, y_train, ax, name)
        else:
            plot_knn_decision_boundary(model, X_train, y_train, ax, name)
    
    plt.tight_layout()
    plt.show()


def plot_logistic_boundary_on_ax(model, X, y, ax, title):
    """Plot logistic regression decision boundary on given axis."""
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)[:, 1]
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


if __name__ == "__main__":
    print("MACHINE LEARNING ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates implementations of classical ML algorithms")
    print("with visualizations and comparisons to scikit-learn.")
    print()
    
    # Run all demonstrations
    linear_regression_demo()
    logistic_regression_demo()
    knn_demo()
    algorithm_comparison()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED!")
    print("All algorithms have been demonstrated with examples and visualizations.")
    print("Check the generated plots for visual insights.")
    print("=" * 60)
