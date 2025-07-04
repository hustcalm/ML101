# Examples

This directory contains comprehensive examples demonstrating the usage of ML101 algorithm implementations. Each example script showcases different algorithms, techniques, and comparison studies.

## Available Examples

### `linear_regression_example.py`
Demonstrates linear regression with both normal equation and gradient descent methods.

**Features:**
- Basic linear regression usage
- Comparison of solving methods
- Visualization of cost function convergence
- Performance evaluation with different noise levels
- Feature scaling demonstration

**Run:**
```bash
python examples/linear_regression_example.py
```

### `classification_comparison.py`
Comprehensive comparison of classification algorithms on various datasets.

**Algorithms Compared:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Naive Bayes
- Support Vector Machines (SVM)

**Features:**
- Multiple dataset evaluations
- Cross-validation analysis
- Performance metrics comparison
- Visualization of decision boundaries
- Hyperparameter sensitivity analysis

**Run:**
```bash
python examples/classification_comparison.py
```

### `advanced_algorithms_demo.py`
Showcases advanced machine learning techniques including ensemble methods and dimensionality reduction.

**Algorithms Demonstrated:**
- Support Vector Machines (SVM) with different kernels
- Principal Component Analysis (PCA)
- Random Forest ensemble method

**Features:**
- SVM kernel comparison (linear, RBF, polynomial, sigmoid)
- PCA dimensionality reduction and visualization
- Random Forest feature importance analysis
- Performance comparison with scikit-learn implementations
- Advanced visualization techniques

**Run:**
```bash
python examples/advanced_algorithms_demo.py
```

### `regularized_regression_demo.py`
Comprehensive comparison of regularized regression techniques.

**Algorithms Compared:**
- Linear Regression (no regularization)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)

**Features:**
- Regularization strength comparison
- Feature selection analysis (Lasso)
- Multicollinearity handling (Ridge)
- Regularization path visualization
- Coefficient shrinkage demonstration
- Bias-variance tradeoff illustration

**Run:**
```bash
python examples/regularized_regression_demo.py
```

## Example Workflows

### Getting Started with Classification
```python
# 1. Import required modules
from algorithms.logistic_regression.logistic_regression import LogisticRegression
from utils.preprocessing import StandardScaler, train_test_split
from utils.metrics import ClassificationMetrics

# 2. Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X_train_scaled, y_train)

# 4. Make predictions and evaluate
y_pred = model.predict(X_test_scaled)
metrics = ClassificationMetrics()
accuracy = metrics.accuracy(y_test, y_pred)
```

### Regression Analysis Pipeline
```python
# 1. Compare different regression methods
from algorithms.linear_regression.linear_regression import LinearRegression
from algorithms.ridge_regression.ridge_regression import RidgeRegression
from algorithms.lasso_regression.lasso_regression import LassoRegression

# 2. Prepare models
models = {
    'Linear': LinearRegression(),
    'Ridge': RidgeRegression(alpha=1.0),
    'Lasso': LassoRegression(alpha=0.1)
}

# 3. Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results[name] = score
    print(f"{name} R² Score: {score:.4f}")
```

### Ensemble Learning Example
```python
# Random Forest demonstration
from algorithms.random_forest.random_forest import RandomForest

# Create and train ensemble
rf = RandomForest(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Analyze feature importance
importances = rf.feature_importances_
top_features = np.argsort(importances)[::-1][:5]
print("Top 5 most important features:", top_features)

# Plot feature importance
rf.plot_feature_importances()
```

## Common Example Patterns

### Data Preparation
All examples follow consistent data preparation patterns:

```python
# 1. Generate or load data
X, y = generate_sample_data(n_samples=1000, n_features=10, random_state=42)

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Model Evaluation
Consistent evaluation approach across all examples:

```python
# 1. Train model
model.fit(X_train, y_train)

# 2. Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 3. Calculate metrics
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# 4. Check for overfitting
if train_score - test_score > 0.1:
    print("Warning: Possible overfitting detected")
```

### Visualization
Examples include comprehensive visualizations:

```python
# 1. Learning curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_sizes, train_scores, 'o-', label='Training')
plt.plot(train_sizes, val_scores, 'o-', label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curve')

# 2. Feature importance
plt.subplot(1, 3, 2)
plt.bar(range(len(importances)), importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')

# 3. Decision boundaries (for 2D data)
plt.subplot(1, 3, 3)
plot_decision_boundary(model, X, y)
plt.title('Decision Boundary')

plt.tight_layout()
plt.show()
```

## Running the Examples

### Prerequisites
Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

### Individual Examples
Run specific examples:
```bash
# Basic linear regression
python examples/linear_regression_example.py

# Classification comparison
python examples/classification_comparison.py

# Advanced algorithms (SVM, PCA, Random Forest)
python examples/advanced_algorithms_demo.py

# Regularized regression comparison
python examples/regularized_regression_demo.py
```

### All Examples
Run all examples in sequence:
```bash
# From the ML101 root directory
for example in examples/*.py; do
    echo "Running $example..."
    python "$example"
    echo "Completed $example"
    echo "---"
done
```

## Customization

### Modifying Parameters
Examples are designed to be easily customizable:

```python
# Example: Modify dataset parameters
X, y = generate_sample_data(
    n_samples=2000,      # Increase sample size
    n_features=20,       # More features
    noise=0.05,          # Less noise
    random_state=123     # Different random seed
)

# Example: Modify model parameters
model = RandomForest(
    n_estimators=200,    # More trees
    max_depth=15,        # Deeper trees
    max_features='log2', # Different feature selection
    random_state=42
)
```

### Adding New Comparisons
Extend examples with additional algorithms:

```python
# Add new algorithm to comparison
from algorithms.new_algorithm.new_algorithm import NewAlgorithm

models = {
    'Existing Algorithm': ExistingAlgorithm(),
    'New Algorithm': NewAlgorithm(param1=value1, param2=value2)
}

# Rest of comparison code remains the same
```

## Educational Value

### Learning Objectives
Each example is designed to teach specific concepts:

1. **Algorithm Implementation**: See how algorithms work internally
2. **Parameter Tuning**: Understand hyperparameter effects
3. **Performance Evaluation**: Learn proper evaluation techniques
4. **Comparative Analysis**: Understand when to use which algorithm
5. **Visualization**: Interpret results through plots and graphs

### Best Practices Demonstrated
- Proper train/validation/test splits
- Feature scaling and preprocessing
- Cross-validation techniques
- Overfitting detection and prevention
- Comprehensive performance evaluation
- Statistical significance testing

### Common Pitfalls Avoided
- Data leakage in preprocessing
- Overfitting through parameter tuning on test set
- Inappropriate metric selection
- Ignoring class imbalance
- Forgetting to scale features

## Integration Testing

Examples also serve as integration tests, ensuring:
- All algorithms work correctly
- Interfaces are consistent
- Visualizations render properly
- Performance is reasonable
- Documentation matches implementation

## Contributing

When adding new examples:

1. **Follow naming convention**: `algorithm_name_example.py` or `comparison_topic.py`
2. **Include comprehensive documentation**: Explain what the example demonstrates
3. **Add visualization**: Include relevant plots and graphs
4. **Demonstrate best practices**: Show proper ML workflow
5. **Test thoroughly**: Ensure example runs without errors
6. **Update this README**: Add description of new example

These examples provide both educational value and practical demonstrations of the ML101 implementations, making them accessible to learners at all levels.
