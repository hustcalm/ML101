# Linear Regression

## Overview

Linear Regression is one of the fundamental supervised learning algorithms used for predicting continuous target variables. It assumes a linear relationship between input features and the target variable.

## Mathematical Foundation

### Simple Linear Regression
For a single feature, the model is:
```
y = β₀ + β₁x + ε
```

Where:
- `y` is the target variable
- `x` is the input feature
- `β₀` is the bias (intercept)
- `β₁` is the weight (slope)
- `ε` is the error term

### Multiple Linear Regression
For multiple features:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

In matrix form:
```
y = Xβ + ε
```

## Solution Methods

### 1. Normal Equation (Analytical Solution)
The optimal parameters can be found directly using:
```
β = (X^T X)^(-1) X^T y
```

**Advantages:**
- Exact solution
- No hyperparameters to tune
- Works well for small to medium datasets

**Disadvantages:**
- Computationally expensive for large datasets (O(n³))
- Requires matrix inversion
- May fail if X^T X is singular

### 2. Gradient Descent (Iterative Solution)
Minimizes the cost function iteratively:

**Cost Function (Mean Squared Error):**
```
J(β) = (1/2m) Σ(h_β(x^(i)) - y^(i))²
```

**Parameter Update Rule:**
```
β := β - α ∇J(β)
```

Where α is the learning rate.

**Advantages:**
- Scales well to large datasets
- Memory efficient
- Can handle singular matrices

**Disadvantages:**
- Requires hyperparameter tuning
- May converge slowly
- Can get stuck in local minima (though MSE is convex)

## Key Assumptions

1. **Linearity**: The relationship between features and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features are not highly correlated

## Implementation Features

Our implementation includes:
- Both Normal Equation and Gradient Descent methods
- Automatic convergence detection
- Cost function tracking
- R² score calculation
- Visualization capabilities

## Usage Example

```python
from linear_regression import LinearRegression, generate_linear_data

# Generate sample data
X, y = generate_linear_data(n_samples=100, noise=0.1, random_state=42)

# Fit using Normal Equation
model_normal = LinearRegression(method='normal')
model_normal.fit(X, y)

# Fit using Gradient Descent
model_gd = LinearRegression(method='gradient_descent', learning_rate=0.01)
model_gd.fit(X, y)

# Make predictions
predictions = model_normal.predict(X)

# Evaluate
r2_score = model_normal.score(X, y)
print(f"R² Score: {r2_score:.4f}")
```

## When to Use Linear Regression

**Good for:**
- Simple, interpretable models
- When relationship is approximately linear
- Baseline models
- Small to medium datasets
- When you need feature importance

**Not ideal for:**
- Non-linear relationships
- High-dimensional data with many irrelevant features
- When target variable is categorical
