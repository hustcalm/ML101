# Ridge Regression

Ridge Regression is a linear regression technique that includes L2 regularization to prevent overfitting. It adds a penalty term proportional to the square of the coefficients to the ordinary least squares cost function.

## Mathematical Foundation

### Cost Function

Ridge regression modifies the ordinary least squares cost function by adding an L2 penalty term:

```
J(θ) = (1/2m) * Σᵢ₌₁ᵐ (hθ(xᵢ) - yᵢ)² + λ * Σⱼ₌₁ⁿ θⱼ²
```

Where:
- First term: Mean Squared Error (same as linear regression)
- Second term: L2 regularization penalty
- λ (lambda/alpha): Regularization strength parameter

### Normal Equation Solution

The closed-form solution for Ridge regression is:

```
θ = (XᵀX + λI)⁻¹Xᵀy
```

Where:
- I is the identity matrix
- λI is added to make the matrix invertible and provide regularization

### Gradient Descent

For gradient descent, the gradient includes the regularization term:

```
∇J = (1/m) * Xᵀ(Xθ - y) + λθ
```

Update rule:
```
θ := θ - α * [(1/m) * Xᵀ(Xθ - y) + λθ]
```

## Key Properties

### Regularization Effect

**L2 Penalty (Ridge):**
- Shrinks coefficients towards zero (but not exactly zero)
- Handles multicollinearity by distributing coefficients among correlated features
- Produces more stable solutions

**Effect of λ (alpha):**
- λ = 0: Reduces to ordinary linear regression
- λ → ∞: All coefficients approach zero
- Optimal λ: Balances bias-variance tradeoff

### Bias-Variance Tradeoff

- **Increasing λ**: Higher bias, lower variance
- **Decreasing λ**: Lower bias, higher variance
- **Goal**: Find λ that minimizes total error = bias² + variance + noise

## Implementation Details

### Solvers

1. **Normal Equation (Closed-form)**
   - Exact solution in one step
   - Computationally expensive for large datasets (O(n³))
   - Always finds global minimum

2. **Gradient Descent**
   - Iterative approach
   - Scales better with large datasets
   - Requires tuning learning rate and iterations

### Important Notes

- **Intercept**: Usually not regularized (regularization matrix has 0 for intercept)
- **Feature Scaling**: Recommended since regularization is sensitive to feature scales
- **Matrix Invertibility**: λI ensures XᵀX + λI is always invertible

## Usage Example

```python
from ridge_regression import RidgeRegression, generate_regression_data
import numpy as np

# Generate sample data
X, y = generate_regression_data(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Ridge Regression with different regularization strengths
alphas = [0.1, 1.0, 10.0]

for alpha in alphas:
    ridge = RidgeRegression(alpha=alpha, solver='normal')
    ridge.fit(X_train, y_train)
    
    train_score = ridge.score(X_train, y_train)
    test_score = ridge.score(X_test, y_test)
    
    print(f"α = {alpha}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

# Compare solvers
ridge_normal = RidgeRegression(alpha=1.0, solver='normal')
ridge_gd = RidgeRegression(alpha=1.0, solver='gradient_descent')

ridge_normal.fit(X_train, y_train)
ridge_gd.fit(X_train, y_train)

# Predictions
predictions = ridge_normal.predict(X_test)

# Visualize regularization path
alphas_path = np.logspace(-3, 3, 50)
ridge_normal.plot_regularization_path(X_train, y_train, alphas_path)
```

## Parameters

- **alpha** (float): Regularization strength. Higher values specify stronger regularization
- **fit_intercept** (bool): Whether to fit an intercept term
- **solver** (str): 'normal' for closed-form solution, 'gradient_descent' for iterative
- **max_iter** (int): Maximum iterations for gradient descent
- **learning_rate** (float): Step size for gradient descent
- **tolerance** (float): Convergence tolerance for gradient descent

## Advantages

1. **Prevents Overfitting**: Regularization reduces model complexity
2. **Handles Multicollinearity**: Distributes weights among correlated features
3. **Stable Solutions**: Always has a unique solution due to regularization
4. **Closed-form Solution**: Exact solution available via normal equation
5. **Interpretable**: Linear model with clear coefficient interpretation

## Disadvantages

1. **Biased Estimates**: Regularization introduces bias
2. **Feature Selection**: Doesn't perform automatic feature selection (coefficients → 0 but ≠ 0)
3. **Hyperparameter Tuning**: Requires selection of optimal λ
4. **Scale Sensitivity**: Performance depends on feature scaling
5. **Less Sparse**: Unlike Lasso, doesn't produce exactly zero coefficients

## When to Use Ridge Regression

### Good Use Cases

1. **Multicollinearity**: When features are highly correlated
2. **Small Sample Size**: When n < p (more features than samples)
3. **Stable Predictions**: When you want to retain all features but shrink coefficients
4. **Interpretability**: When you need a linear, interpretable model

### Compare with Other Methods

| Method | Regularization | Feature Selection | Sparse Solutions |
|--------|---------------|-------------------|------------------|
| Linear Regression | None | No | No |
| Ridge Regression | L2 | No | No |
| Lasso Regression | L1 | Yes | Yes |
| Elastic Net | L1 + L2 | Yes | Partial |

## Hyperparameter Tuning

### Cross-Validation for α

```python
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-3, 3, 100)
best_alpha = None
best_score = -np.inf

for alpha in alphas:
    ridge = RidgeRegression(alpha=alpha)
    scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
    mean_score = np.mean(scores)
    
    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

print(f"Best α: {best_alpha:.4f}, Best CV Score: {best_score:.4f}")
```

### Grid Search Strategy

1. Start with a wide range: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
2. Narrow down around the best performing region
3. Use logarithmic spacing for efficiency
4. Consider computational cost vs. performance gain

## Mathematical Intuition

### Geometric Interpretation

Ridge regression can be viewed as constrained optimization:

```
minimize: ||Xθ - y||₂²
subject to: ||θ||₂² ≤ t
```

The L2 constraint creates a circular (spherical in higher dimensions) constraint region. The solution is where the elliptical contours of the objective function first touch this circular constraint.

### Bayesian Interpretation

Ridge regression is equivalent to Maximum A Posteriori (MAP) estimation with a Gaussian prior on coefficients:

```
θ ~ N(0, σ²/λ * I)
```

The regularization parameter λ is inversely related to the prior variance.

## Practical Tips

### 1. Feature Scaling
Always scale features when using Ridge regression:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Choosing α
- Start with α = 1.0 as a reasonable default
- Use cross-validation to find optimal value
- Plot validation curves to understand bias-variance tradeoff

### 3. Regularization Path
Examine how coefficients change with α:
```python
ridge.plot_regularization_path(X_train, y_train, alphas)
```

### 4. Solver Selection
- Use 'normal' for small to medium datasets (< 10,000 samples)
- Use 'gradient_descent' for large datasets or when memory is limited

## Extensions and Variants

### 1. Kernel Ridge Regression
Extend to non-linear relationships using kernel trick

### 2. Bayesian Ridge Regression
Automatic relevance determination with uncertainty quantification

### 3. Multi-task Ridge Regression
Share information across related regression tasks

### 4. Ridge Classification
Apply Ridge penalty to logistic regression for classification

## Comparison with Linear Regression

| Aspect | Linear Regression | Ridge Regression |
|--------|------------------|------------------|
| **Bias** | Unbiased | Biased (shrinkage) |
| **Variance** | High (with multicollinearity) | Lower |
| **Overfitting** | Prone to overfitting | Reduced overfitting |
| **Multicollinearity** | Unstable | Stable |
| **Feature Selection** | No | No (shrinkage only) |
| **Interpretability** | High | High |
| **Computational Cost** | Lower | Slightly higher |

Ridge regression is particularly valuable when dealing with multicollinearity or when you have more features than samples, providing a more stable and generalizable solution than ordinary linear regression.
