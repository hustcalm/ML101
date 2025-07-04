# Utility Functions

This directory contains utility functions and helper classes used across the ML101 project implementations.

## Available Modules

### `metrics.py`
Comprehensive evaluation metrics for both classification and regression tasks.

#### Classification Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs actual labels
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Classification Report**: Complete summary of all metrics

#### Regression Metrics
- **Mean Squared Error (MSE)**: Average of squared differences
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Error (MAE)**: Average of absolute differences
- **R² Score**: Coefficient of determination
- **Adjusted R²**: R² adjusted for number of features
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric

### `preprocessing.py`
Data preprocessing and transformation utilities.

#### Scalers
- **StandardScaler**: Standardizes features by removing mean and scaling to unit variance
- **MinMaxScaler**: Scales features to a given range (default [0, 1])
- **RobustScaler**: Uses median and IQR for robust scaling

#### Encoders
- **LabelEncoder**: Encodes categorical labels with values 0 to n_classes-1
- **OneHotEncoder**: Converts categorical variables into binary vectors

#### Data Splitting
- **train_test_split**: Splits dataset into training and testing sets
- **train_val_test_split**: Splits dataset into training, validation, and testing sets

#### Feature Engineering
- **PolynomialFeatures**: Generates polynomial and interaction features
- **feature_selection**: Basic feature selection utilities

## Usage Examples

### Classification Metrics
```python
from utils.metrics import ClassificationMetrics
import numpy as np

# Generate sample predictions
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])

# Calculate metrics
metrics = ClassificationMetrics()
accuracy = metrics.accuracy(y_true, y_pred)
precision = metrics.precision(y_true, y_pred)
recall = metrics.recall(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Generate full report
report = metrics.classification_report(y_true, y_pred)
print("Classification Report:")
print(report)
```

### Regression Metrics
```python
from utils.metrics import RegressionMetrics
import numpy as np

# Generate sample data
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.8, 3.9, 5.1])

# Calculate metrics
metrics = RegressionMetrics()
mse = metrics.mean_squared_error(y_true, y_pred)
rmse = metrics.root_mean_squared_error(y_true, y_pred)
mae = metrics.mean_absolute_error(y_true, y_pred)
r2 = metrics.r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
```

### Data Preprocessing
```python
from utils.preprocessing import StandardScaler, train_test_split
import numpy as np

# Generate sample data
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Original data shape: {X.shape}")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training data mean: {np.mean(X_train_scaled, axis=0)}")
print(f"Training data std: {np.std(X_train_scaled, axis=0)}")
```

### Feature Engineering
```python
from utils.preprocessing import PolynomialFeatures
import numpy as np

# Generate sample data
X = np.random.randn(50, 2)

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")
print(f"Feature names: {poly.get_feature_names()}")
```

## Design Principles

### Scikit-learn Compatible API
All utilities follow the scikit-learn API conventions:
- `fit()`: Learn parameters from training data
- `transform()`: Apply transformation to data
- `fit_transform()`: Fit and transform in one step
- `predict()`: Make predictions (for metrics)

### Robust Error Handling
- Input validation for all functions
- Meaningful error messages
- Graceful handling of edge cases

### Comprehensive Documentation
- Detailed docstrings for all functions
- Mathematical formulations where applicable
- Usage examples and best practices

### Efficient Implementation
- Vectorized operations using NumPy
- Memory-efficient algorithms
- Optimized for common use cases

## Best Practices

### Data Preprocessing Pipeline
1. **Split data first**: Always split before any preprocessing
2. **Fit on training data only**: Never fit scalers on test data
3. **Apply same transformation**: Use fitted transformers on test data
4. **Handle missing values**: Deal with NaN values before scaling
5. **Feature selection**: Consider feature importance and correlation

### Evaluation Strategy
1. **Hold-out validation**: Use separate test set for final evaluation
2. **Cross-validation**: Use k-fold CV for model selection
3. **Multiple metrics**: Don't rely on a single metric
4. **Baseline comparison**: Always compare against simple baselines
5. **Statistical significance**: Test if improvements are statistically significant

### Common Pitfalls to Avoid
1. **Data leakage**: Don't fit preprocessors on test data
2. **Overfitting**: Don't tune hyperparameters on test set
3. **Unbalanced datasets**: Use appropriate metrics (precision, recall, F1)
4. **Scale sensitivity**: Remember to scale features for distance-based algorithms
5. **Feature correlation**: Check for multicollinearity in linear models

## Integration with ML101 Algorithms

All ML101 algorithm implementations use these utilities:

```python
# Example workflow
from algorithms.linear_regression.linear_regression import LinearRegression
from utils.preprocessing import StandardScaler, train_test_split
from utils.metrics import RegressionMetrics

# Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
metrics = RegressionMetrics()
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")
```

This utility module provides the foundation for consistent, reliable machine learning workflows across all ML101 implementations.
