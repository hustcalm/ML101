# Lasso Regression

Lasso (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that includes L1 regularization. Unlike Ridge regression, Lasso can drive coefficients to exactly zero, performing automatic feature selection.

## Mathematical Foundation

### Cost Function

Lasso regression modifies the ordinary least squares cost function by adding an L1 penalty term:

```
J(θ) = (1/2m) * Σᵢ₌₁ᵐ (hθ(xᵢ) - yᵢ)² + λ * Σⱼ₌₁ⁿ |θⱼ|
```

Where:
- First term: Mean Squared Error (same as linear regression)
- Second term: L1 regularization penalty
- λ (lambda/alpha): Regularization strength parameter

### Key Differences from Ridge

**Ridge (L2):** `λ * Σθⱼ²` → Shrinks coefficients toward zero
**Lasso (L1):** `λ * Σ|θⱼ|` → Sets some coefficients to exactly zero

### No Closed-Form Solution

Unlike Ridge regression, Lasso has no closed-form solution due to the non-differentiable L1 penalty at θ = 0. Common algorithms include:

1. **Coordinate Descent**: Most common, updates one coefficient at a time
2. **Proximal Gradient**: Uses soft thresholding operator
3. **LARS (Least Angle Regression)**: Efficient path algorithm

### Soft Thresholding Operator

The soft thresholding operator is key to solving Lasso:

```
soft_threshold(x, λ) = {
    x - λ    if x > λ
    x + λ    if x < -λ
    0        if |x| ≤ λ
}
```

## Implementation: Coordinate Descent

The coordinate descent algorithm updates one coefficient at a time:

1. **Initialize**: θ = 0
2. **For each iteration**:
   - For each coefficient j:
     - Compute partial residual: r⁽ʲ⁾ = y - Σₖ≠ⱼ Xₖθₖ
     - Update: θⱼ = soft_threshold(Xⱼᵀr⁽ʲ⁾/||Xⱼ||², λ/||Xⱼ||²)
3. **Repeat** until convergence

## Key Properties

### Automatic Feature Selection

**Sparsity**: L1 penalty drives coefficients to exactly zero
- **λ = 0**: All features retained (linear regression)
- **λ → ∞**: All coefficients → 0 (null model)
- **Optimal λ**: Balance between fit and sparsity

### Regularization Path

As λ increases, coefficients are eliminated in order of importance:
1. Least important features go to zero first
2. Most important features remain longer
3. Creates natural feature ranking

### Geometric Interpretation

Lasso can be viewed as constrained optimization:

```
minimize: ||Xθ - y||₂²
subject to: ||θ||₁ ≤ t
```

The L1 constraint creates a diamond-shaped (simplicial in higher dimensions) constraint region. Solutions tend to occur at corners where some coordinates are zero.

## Usage Example

```python
from lasso_regression import LassoRegression, generate_sparse_regression_data
import numpy as np

# Generate sparse data
X, y, true_coef = generate_sparse_regression_data(
    n_samples=100, n_features=20, n_informative=5, 
    noise=0.1, random_state=42
)

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Lasso with different regularization strengths
alphas = [0.001, 0.01, 0.1, 1.0]

for alpha in alphas:
    lasso = LassoRegression(alpha=alpha, max_iter=1000)
    lasso.fit(X_train, y_train)
    
    train_score = lasso.score(X_train, y_train)
    test_score = lasso.score(X_test, y_test)
    n_selected = len(lasso.get_selected_features())
    
    print(f"α = {alpha}: Train R² = {train_score:.4f}, "
          f"Test R² = {test_score:.4f}, Features = {n_selected}")

# Feature selection analysis
selected_features = lasso.get_selected_features()
print(f"Selected features: {selected_features}")

# Visualize regularization path
alphas_path = np.logspace(-3, 1, 50)
lasso.plot_regularization_path(X_train, y_train, alphas_path)
```

## Parameters

- **alpha** (float): Regularization strength. Higher values create sparser models
- **fit_intercept** (bool): Whether to fit an intercept term
- **max_iter** (int): Maximum iterations for coordinate descent
- **tolerance** (float): Convergence tolerance
- **positive** (bool): Force coefficients to be positive

## Advantages

1. **Automatic Feature Selection**: Eliminates irrelevant features
2. **Interpretable Models**: Sparse solutions are easier to interpret
3. **Handles High-Dimensional Data**: Works when p > n
4. **Reduces Overfitting**: Regularization improves generalization
5. **Variable Selection**: Identifies most important features

## Disadvantages

1. **Arbitrary Selection**: Among correlated features, picks one arbitrarily
2. **Grouping Effect**: Doesn't group correlated features (unlike Elastic Net)
3. **Instability**: Small data changes can lead to different feature selections
4. **Bias**: Can eliminate truly important features with high λ
5. **No Closed-Form**: Requires iterative algorithms

## When to Use Lasso Regression

### Good Use Cases

1. **Feature Selection**: When you need to identify important features
2. **High-Dimensional Data**: When you have many features (p >> n)
3. **Interpretable Models**: When model simplicity is important
4. **Sparse Ground Truth**: When true model is known to be sparse

### Avoid When

1. **Grouped Features**: When correlated features should be selected together
2. **All Features Important**: When most features are relevant
3. **Small Sample Sizes**: Can be unstable with very small datasets

## Hyperparameter Tuning

### Cross-Validation for α

```python
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-4, 1, 100)
cv_scores = []

for alpha in alphas:
    lasso = LassoRegression(alpha=alpha)
    scores = cross_val_score(lasso, X_train, y_train, cv=5, scoring='r2')
    cv_scores.append(np.mean(scores))

best_alpha = alphas[np.argmax(cv_scores)]
print(f"Best α: {best_alpha:.4f}")
```

### Information Criteria

Use AIC/BIC to balance fit and model complexity:

```python
def aic_score(y_true, y_pred, n_params):
    mse = np.mean((y_true - y_pred) ** 2)
    return len(y_true) * np.log(mse) + 2 * n_params

def bic_score(y_true, y_pred, n_params):
    mse = np.mean((y_true - y_pred) ** 2)
    return len(y_true) * np.log(mse) + n_params * np.log(len(y_true))
```

## Comparison with Other Methods

| Method | Regularization | Feature Selection | Sparse Solutions | Grouped Features |
|--------|----------------|-------------------|------------------|------------------|
| Linear Regression | None | No | No | N/A |
| Ridge Regression | L2 | No | No | Keeps all |
| **Lasso Regression** | **L1** | **Yes** | **Yes** | **Arbitrary** |
| Elastic Net | L1 + L2 | Yes | Yes | Groups similar |

## Advanced Topics

### 1. Regularization Path

The entire solution path can be computed efficiently:

```python
# Plot how coefficients change with λ
alphas = np.logspace(-3, 1, 100)
lasso.plot_regularization_path(X_train, y_train, alphas)
```

### 2. Degrees of Freedom

For Lasso, degrees of freedom ≈ number of non-zero coefficients:
```
df(λ) ≈ |{j : θⱼ(λ) ≠ 0}|
```

### 3. LARS Algorithm

Least Angle Regression computes the entire regularization path efficiently:
- Starts with empty model
- Adds features in order of correlation with residual
- Computes exact λ values where features enter/leave

### 4. Group Lasso

Extension for grouped feature selection:
```
J(θ) = ||Xθ - y||₂² + λ * Σ_g ||θ_g||₂
```

Where θ_g represents coefficients for group g.

## Practical Tips

### 1. Feature Scaling
Always scale features for Lasso:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Choosing α
- Start with α = 1.0
- Use cross-validation with logarithmic grid
- Consider both prediction accuracy and sparsity

### 3. Stability
For stable feature selection:
- Use cross-validation with multiple folds
- Consider Bootstrap Lasso or Stability Selection
- Examine regularization path for sensitivity

### 4. Warm Starts
When trying multiple α values:
```python
# Use previous solution as initialization
lasso = LassoRegression(alpha=0.1)
lasso.fit(X, y)
# Use lasso.coef_ as starting point for next α
```

## Mathematical Insights

### Bayesian Interpretation

Lasso corresponds to MAP estimation with Laplace prior:
```
θⱼ ~ Laplace(0, σ²/λ)
```

The Laplace distribution has a sharp peak at zero, encouraging sparsity.

### Subdifferential

Since |x| is not differentiable at x = 0, Lasso uses subdifferentials:

```
∂|θⱼ| = {
    1           if θⱼ > 0
    -1          if θⱼ < 0
    [-1, 1]     if θⱼ = 0
}
```

### KKT Conditions

Optimality conditions for Lasso:
1. If θⱼ ≠ 0: Xⱼᵀ(y - Xθ) = λ * sign(θⱼ)
2. If θⱼ = 0: |Xⱼᵀ(y - Xθ)| ≤ λ

## Extensions and Variants

### 1. Adaptive Lasso
Weights features differently:
```
J(θ) = ||Xθ - y||₂² + λ * Σⱼ wⱼ|θⱼ|
```

### 2. Fused Lasso
Encourages sparsity in differences:
```
J(θ) = ||Xθ - y||₂² + λ₁||θ||₁ + λ₂||Dθ||₁
```

### 3. Elastic Net
Combines L1 and L2 penalties:
```
J(θ) = ||Xθ - y||₂² + λ₁||θ||₁ + λ₂||θ||₂²
```

### 4. Multi-task Lasso
Feature selection across related tasks:
```
J(Θ) = ||XΘ - Y||²_F + λ * Σⱼ ||θⱼ||₂
```

## Comparison with Ridge Regression

| Aspect | Ridge | Lasso |
|--------|--------|--------|
| **Penalty** | L2: Σθ² | L1: Σ\|θ\| |
| **Solution** | Closed-form | Iterative |
| **Feature Selection** | No | Yes |
| **Sparsity** | No | Yes |
| **Grouped Features** | Keeps all | Picks one |
| **Stability** | High | Lower |
| **Interpretability** | Good | Excellent |

## Performance Considerations

### Computational Complexity
- **Coordinate Descent**: O(knp) where k is iterations
- **Path Algorithms**: O(kp³) for full path
- **Memory**: O(np) for data storage

### Convergence
- Usually converges quickly
- Can be slow near optimum
- Warm starts help with multiple λ values

Lasso regression is particularly powerful for feature selection and creating interpretable models, making it a valuable tool in the machine learning toolkit when sparsity and interpretability are important.
