"""
Linear Regression Example

This script demonstrates the Linear Regression implementation with:
1. Synthetic data generation
2. Comparison between Normal Equation and Gradient Descent
3. Visualization of results
4. Comparison with scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from ml101 import LinearRegression
from ml101.utils import train_test_split

# Generate synthetic data
def generate_linear_data(n_samples=100, n_features=1, noise=0.1, random_state=None):
    """Generate synthetic linear regression data."""
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + np.random.randn(n_samples) * noise
    
    return X, y


def compare_methods():
    """Compare Normal Equation vs Gradient Descent methods."""
    print("=== Linear Regression: Method Comparison ===\n")
    
    # Generate sample data
    X, y = generate_linear_data(n_samples=100, noise=0.2, random_state=42)
    
    # Fit using Normal Equation
    print("1. Normal Equation Method")
    model_normal = LinearRegression(method='normal')
    model_normal.fit(X, y)
    
    # Fit using Gradient Descent
    print("2. Gradient Descent Method")
    model_gd = LinearRegression(method='gradient_descent', learning_rate=0.01, max_iterations=1000)
    model_gd.fit(X, y)
    
    # Compare results
    print(f"\nResults Comparison:")
    print(f"Normal Equation - Weight: {model_normal.weights[0]:.4f}, Bias: {model_normal.bias:.4f}")
    print(f"Gradient Descent - Weight: {model_gd.weights[0]:.4f}, Bias: {model_gd.bias:.4f}")
    
    # R² scores
    r2_normal = model_normal.score(X, y)
    r2_gd = model_gd.score(X, y)
    print(f"\nR² Scores:")
    print(f"Normal Equation: {r2_normal:.4f}")
    print(f"Gradient Descent: {r2_gd:.4f}")
    
    return model_normal, model_gd, X, y


def visualize_results(model_normal, model_gd, X, y):
    """Visualize the regression results."""
    print("\n=== Visualization ===")
    
    # Create test points for smooth line
    X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred_normal = model_normal.predict(X_test)
    y_pred_gd = model_gd.predict(X_test)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Normal Equation Results
    axes[0, 0].scatter(X, y, alpha=0.6, color='blue', label='Data points')
    axes[0, 0].plot(X_test, y_pred_normal, color='red', linewidth=2, label='Normal Equation')
    axes[0, 0].set_title('Linear Regression - Normal Equation')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Gradient Descent Results
    axes[0, 1].scatter(X, y, alpha=0.6, color='blue', label='Data points')
    axes[0, 1].plot(X_test, y_pred_gd, color='green', linewidth=2, label='Gradient Descent')
    axes[0, 1].set_title('Linear Regression - Gradient Descent')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cost History for Gradient Descent
    if model_gd.cost_history:
        axes[1, 0].plot(model_gd.cost_history, color='purple')
        axes[1, 0].set_title('Cost Function During Training (Gradient Descent)')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Mean Squared Error')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals Analysis
    y_pred_normal_train = model_normal.predict(X)
    residuals = y - y_pred_normal_train
    axes[1, 1].scatter(y_pred_normal_train, residuals, alpha=0.6, color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_title('Residuals Plot')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_with_sklearn(X, y):
    """Compare our implementation with scikit-learn."""
    print("\n=== Comparison with Scikit-learn ===")
    
    # Our implementation
    our_model = LinearRegression(method='normal')
    our_model.fit(X, y)
    our_predictions = our_model.predict(X)
    our_r2 = our_model.score(X, y)
    
    # Scikit-learn implementation
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X, y)
    sklearn_predictions = sklearn_model.predict(X)
    sklearn_r2 = r2_score(y, sklearn_predictions)
    
    print(f"Our Implementation:")
    print(f"  Weight: {our_model.weights[0]:.6f}")
    print(f"  Bias: {our_model.bias:.6f}")
    print(f"  R² Score: {our_r2:.6f}")
    
    print(f"\nScikit-learn:")
    print(f"  Weight: {sklearn_model.coef_[0]:.6f}")
    print(f"  Bias: {sklearn_model.intercept_:.6f}")
    print(f"  R² Score: {sklearn_r2:.6f}")
    
    print(f"\nDifference in parameters:")
    print(f"  Weight difference: {abs(our_model.weights[0] - sklearn_model.coef_[0]):.8f}")
    print(f"  Bias difference: {abs(our_model.bias - sklearn_model.intercept_):.8f}")


def multivariable_example():
    """Demonstrate multivariable linear regression."""
    print("\n=== Multivariable Linear Regression ===")
    
    # Generate multivariable data
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.5, -2.0, 0.8])
    true_bias = 0.5
    noise = np.random.normal(0, 0.1, n_samples)
    y = X @ true_weights + true_bias + noise
    
    # Fit our model
    model = LinearRegression(method='normal')
    model.fit(X, y)
    
    print(f"True parameters:")
    print(f"  Weights: {true_weights}")
    print(f"  Bias: {true_bias}")
    
    print(f"\nEstimated parameters:")
    print(f"  Weights: {model.weights}")
    print(f"  Bias: {model.bias:.4f}")
    
    print(f"\nR² Score: {model.score(X, y):.4f}")


def learning_rate_analysis():
    """Analyze the effect of different learning rates."""
    print("\n=== Learning Rate Analysis ===")
    
    X, y = generate_linear_data(n_samples=100, noise=0.1, random_state=42)
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        model = LinearRegression(method='gradient_descent', 
                               learning_rate=lr, 
                               max_iterations=1000)
        model.fit(X, y)
        
        plt.subplot(2, 2, i+1)
        plt.plot(model.cost_history)
        plt.title(f'Learning Rate: {lr}')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
        
        print(f"Learning Rate {lr}: Final cost = {model.cost_history[-1]:.6f}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Main demonstration
    print("Linear Regression Implementation Demo")
    print("=" * 50)
    
    # Compare methods
    model_normal, model_gd, X, y = compare_methods()
    
    # Visualize results
    visualize_results(model_normal, model_gd, X, y)
    
    # Compare with sklearn
    compare_with_sklearn(X, y)
    
    # Multivariable example
    multivariable_example()
    
    # Learning rate analysis
    learning_rate_analysis()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the generated plots.")
