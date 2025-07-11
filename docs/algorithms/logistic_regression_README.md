# Logistic Regression

## Overview

Logistic Regression is a linear classifier used for binary and multiclass classification problems. Despite its name, it's a classification algorithm that uses the logistic function to model the probability that an instance belongs to a particular class.

## Mathematical Foundation

### Binary Classification

The logistic regression model predicts the probability that an instance belongs to the positive class:

```
P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
```

Where:
- `z = w^T x + b` (linear combination)
- `σ(z)` is the sigmoid function
- `w` are the weights
- `b` is the bias

### Decision Boundary
The decision boundary is where P(y=1|x) = 0.5, which occurs when z = 0:
```
w^T x + b = 0
```

### Cost Function
Logistic regression uses the logistic loss (cross-entropy):
```
J(w) = -(1/m) Σ [y^(i) log(h(x^(i))) + (1-y^(i)) log(1-h(x^(i)))]
```

### Gradient Descent
The parameter updates are:
```
w := w - α ∂J/∂w
b := b - α ∂J/∂b
```

Where the gradients are:
```
∂J/∂w = (1/m) X^T (h(X) - y)
∂J/∂b = (1/m) Σ (h(x^(i)) - y^(i))
```

## Multiclass Classification

For multiclass problems, we use the One-vs-Rest (OvR) approach:
1. Train one binary classifier for each class
2. For prediction, choose the class with highest probability

Alternatively, multinomial logistic regression uses softmax:
```
P(y=k|x) = exp(z_k) / Σ exp(z_j)
```

## Regularization

### L1 Regularization (Lasso)
Adds penalty: `λ Σ |w_i|`
- Promotes sparsity
- Automatic feature selection

### L2 Regularization (Ridge)
Adds penalty: `λ Σ w_i²`
- Prevents overfitting
- Shrinks weights towards zero

## Key Assumptions

1. **Linear relationship**: Between features and log-odds
2. **Independence**: Observations are independent
3. **No multicollinearity**: Features shouldn't be highly correlated
4. **Large sample size**: For stable results

## Advantages

- **Probabilistic output**: Provides probability estimates
- **No tuning of hyperparameters**: Unlike KNN or SVM
- **Fast training and prediction**: Linear complexity
- **Interpretable**: Coefficients show feature importance
- **No assumptions about feature distributions**

## Disadvantages

- **Linear decision boundary**: Cannot capture complex relationships
- **Sensitive to outliers**: Can affect the sigmoid function
- **Requires large sample sizes**: For stable results
- **Feature scaling**: Performance can be affected by feature scales

## Implementation Features

Our implementation includes:
- Binary and multiclass classification
- Regularization options (L1, L2)
- Convergence detection
- Probability estimates
- Cost function tracking

## Usage Example

```python
from logistic_regression import LogisticRegression, generate_classification_data

# Generate sample data
X, y = generate_classification_data(n_samples=200, n_classes=2, random_state=42)

# Create and train model
model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Evaluate
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

## When to Use Logistic Regression

**Good for:**
- Binary classification problems
- When you need probability estimates
- Baseline models
- Linear decision boundaries
- Interpretable models

**Not ideal for:**
- Non-linear relationships
- Very small datasets
- When features have very different scales (without preprocessing)
