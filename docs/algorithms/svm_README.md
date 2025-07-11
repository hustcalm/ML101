# Support Vector Machine (SVM)

Support Vector Machines are powerful supervised learning models used for classification and regression tasks. This implementation focuses on binary classification using the Sequential Minimal Optimization (SMO) algorithm.

## Mathematical Foundation

### The SVM Optimization Problem

SVM aims to find the optimal hyperplane that separates classes with maximum margin. The optimization problem is:

**Primal Problem:**
```
minimize: ½||w||² + C∑ξᵢ
subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

**Dual Problem:**
```
maximize: ∑αᵢ - ½∑∑αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
subject to: ∑αᵢyᵢ = 0, 0 ≤ αᵢ ≤ C
```

### Decision Function

The decision function for a new sample x is:
```
f(x) = sign(∑αᵢyᵢK(xᵢ,x) + b)
```

Where:
- αᵢ are Lagrange multipliers
- yᵢ are class labels (-1 or +1)
- K(xᵢ,x) is the kernel function
- b is the bias term

### Kernel Functions

1. **Linear Kernel**: K(x₁,x₂) = x₁ᵀx₂
2. **RBF Kernel**: K(x₁,x₂) = exp(-γ||x₁-x₂||²)
3. **Polynomial Kernel**: K(x₁,x₂) = (γx₁ᵀx₂ + r)ᵈ
4. **Sigmoid Kernel**: K(x₁,x₂) = tanh(γx₁ᵀx₂ + r)

## Implementation Details

### Sequential Minimal Optimization (SMO)

SMO algorithm breaks the large quadratic programming problem into smaller problems:

1. **Select two alphas** that violate KKT conditions
2. **Optimize the pair** analytically
3. **Update the bias** term
4. **Repeat** until convergence

### KKT Conditions

For optimal solution, the following must hold:
- αᵢ = 0 ⟹ yᵢfᵢ ≥ 1
- 0 < αᵢ < C ⟹ yᵢfᵢ = 1
- αᵢ = C ⟹ yᵢfᵢ ≤ 1

## Usage Example

```python
from svm import SVM, generate_classification_data
import numpy as np

# Generate sample data
X, y = generate_classification_data(n_samples=200, random_state=42)

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Create and train SVM
svm = SVM(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Make predictions
predictions = svm.predict(X_test)
probabilities = svm.predict_proba(X_test)

# Evaluate
accuracy = svm.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print(f"Support vectors: {svm.n_support_}")

# Visualize decision boundary (for 2D data)
svm.plot_decision_boundary(X_train, y_train)
```

## Parameters

- **C**: Regularization parameter (higher = less regularization)
- **kernel**: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
- **degree**: Polynomial kernel degree
- **gamma**: Kernel coefficient
- **coef0**: Independent term in kernel function
- **tolerance**: Stopping criterion tolerance
- **max_iter**: Maximum iterations
- **random_state**: Random seed for reproducibility

## Advantages

1. **Effective in high dimensions**: Works well with many features
2. **Memory efficient**: Uses only support vectors
3. **Versatile**: Different kernel functions for non-linear problems
4. **Robust**: Less prone to overfitting in high dimensions

## Disadvantages

1. **Slow on large datasets**: O(n³) complexity
2. **Sensitive to feature scaling**: Requires preprocessing
3. **No probability estimates**: Needs additional calibration
4. **Parameter sensitive**: Requires careful tuning

## Key Concepts

### Support Vectors
- Training samples that lie on the margin boundaries
- Only support vectors affect the decision boundary
- Typically small subset of training data

### Margin
- Distance between decision boundary and nearest samples
- SVM maximizes this margin
- Larger margin = better generalization

### Regularization Parameter C
- Controls trade-off between margin and misclassification
- High C: Hard margin (may overfit)
- Low C: Soft margin (may underfit)

## Complexity Analysis

- **Training**: O(n³) for SMO algorithm
- **Prediction**: O(n_sv × n_features) where n_sv is number of support vectors
- **Memory**: O(n_sv × n_features) for storing support vectors

## Common Applications

1. **Text Classification**: Document categorization, spam detection
2. **Image Classification**: Face recognition, object detection
3. **Bioinformatics**: Gene classification, protein folding
4. **Finance**: Credit scoring, fraud detection

## Tips for Better Performance

1. **Scale features**: Use StandardScaler or MinMaxScaler
2. **Tune hyperparameters**: Use cross-validation for C and gamma
3. **Choose appropriate kernel**: Start with RBF, try linear for high dimensions
4. **Handle class imbalance**: Use class_weight parameter
5. **Consider feature selection**: Remove irrelevant features

## Comparison with Other Algorithms

| Algorithm | Pros | Cons |
|-----------|------|------|
| SVM | High accuracy, robust to overfitting | Slow training, needs scaling |
| Logistic Regression | Fast, probabilistic output | Assumes linear relationship |
| Decision Tree | Interpretable, handles non-linear | Prone to overfitting |
| KNN | Simple, no assumptions | Sensitive to irrelevant features |

## Mathematical Intuition

The SVM finds the hyperplane that maximally separates the classes. Think of it as finding the "street" with maximum width that separates two groups of points. The support vectors are the points that lie on the edges of this street - they're the critical points that determine the boundary.

The kernel trick allows SVM to find non-linear boundaries by mapping data to higher dimensions where linear separation becomes possible.
