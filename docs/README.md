# ML101 Documentation

## Project Overview

ML101 is a comprehensive educational project that implements classical machine learning algorithms from scratch in Python. The project is designed to help you understand the mathematical foundations and practical implementation details of fundamental ML algorithms.

## 🎯 Learning Objectives

After completing this project, you will:

- Understand the mathematical foundations of classical ML algorithms
- Be able to implement algorithms from scratch using only NumPy
- Know when to use different algorithms for different types of problems
- Understand the importance of data preprocessing and feature engineering
- Be familiar with proper model evaluation techniques

## 📚 Implemented Algorithms

### Supervised Learning

#### 1. Linear Regression
- **File**: `algorithms/linear_regression/linear_regression.py`
- **Methods**: Normal Equation, Gradient Descent
- **Use Case**: Predicting continuous target variables
- **Key Features**:
  - Both analytical and iterative solutions
  - Convergence detection for gradient descent
  - Cost function visualization

#### 2. Logistic Regression
- **File**: `algorithms/logistic_regression/logistic_regression.py`
- **Methods**: Gradient Descent with sigmoid activation
- **Use Case**: Binary and multiclass classification
- **Key Features**:
  - Probabilistic output
  - Regularization options (L1, L2)
  - One-vs-Rest for multiclass

#### 3. K-Nearest Neighbors (KNN)
- **File**: `algorithms/knn/knn.py`
- **Methods**: Instance-based learning
- **Use Case**: Classification and regression with non-linear decision boundaries
- **Key Features**:
  - Multiple distance metrics (Euclidean, Manhattan, Minkowski)
  - Weighted and uniform voting
  - Works for both classification and regression

### Unsupervised Learning

*More algorithms will be added in future versions*

## 🛠️ Utilities

### Metrics (`utils/metrics.py`)
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression Metrics**: MSE, RMSE, MAE, R², Adjusted R²
- **Visualization**: Confusion matrix plots, regression result plots

### Preprocessing (`utils/preprocessing.py`)
- **Scalers**: StandardScaler, MinMaxScaler
- **Encoders**: LabelEncoder, OneHotEncoder
- **Feature Engineering**: PolynomialFeatures
- **Data Splitting**: train_test_split

## 📁 Project Structure

```
ML101/
├── algorithms/                 # Algorithm implementations
│   ├── linear_regression/
│   │   ├── linear_regression.py
│   │   └── README.md
│   ├── logistic_regression/
│   │   ├── logistic_regression.py
│   │   └── README.md
│   └── knn/
│       ├── knn.py
│       └── README.md
├── examples/                   # Runnable examples
│   ├── linear_regression_example.py
│   └── classification_comparison.py
├── notebooks/                  # Jupyter notebooks
│   └── ml101_tutorial.ipynb
├── utils/                      # Utility functions
│   ├── metrics.py
│   └── preprocessing.py
├── tests/                      # Unit tests
│   └── test_algorithms.py
├── docs/                       # Documentation
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### Installation

1. **Clone the repository** (if using git):
   ```bash
   git clone <repository-url>
   cd ML101
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Examples

1. **Linear Regression Example**:
   ```bash
   python examples/linear_regression_example.py
   ```

2. **Classification Comparison**:
   ```bash
   python examples/classification_comparison.py
   ```

3. **Interactive Jupyter Notebook**:
   ```bash
   jupyter lab notebooks/ml101_tutorial.ipynb
   ```

### Running Tests

```bash
python -m pytest tests/ -v
```

## 📖 Algorithm Details

### Linear Regression

**Mathematical Foundation**:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

**Normal Equation**:
```
β = (X^T X)^(-1) X^T y
```

**Gradient Descent**:
```
β := β - α ∇J(β)
```

**When to Use**:
- Continuous target variable
- Linear relationship between features and target
- Small to medium datasets (Normal Equation)
- Large datasets (Gradient Descent)

### Logistic Regression

**Mathematical Foundation**:
```
P(y=1|x) = σ(w^T x + b) = 1 / (1 + e^(-(w^T x + b)))
```

**Cost Function**:
```
J(w) = -(1/m) Σ [y log(h(x)) + (1-y) log(1-h(x))]
```

**When to Use**:
- Binary or multiclass classification
- Need probability estimates
- Linear decision boundary is appropriate
- Interpretable model required

### K-Nearest Neighbors

**Algorithm**:
1. Calculate distance from query point to all training points
2. Select k nearest neighbors
3. For classification: majority vote; for regression: average

**Distance Metrics**:
- Euclidean: `√(Σ(xi - yi)²)`
- Manhattan: `Σ|xi - yi|`
- Minkowski: `(Σ|xi - yi|^p)^(1/p)`

**When to Use**:
- Non-linear decision boundaries
- No assumptions about data distribution
- Small to medium datasets
- Local patterns in data

## 🎨 Visualization Features

The project includes comprehensive visualization capabilities:

1. **Decision Boundaries**: Visualize how algorithms separate different classes
2. **Learning Curves**: Track cost function during training
3. **Residual Plots**: Analyze regression model performance
4. **Confusion Matrices**: Evaluate classification results
5. **Feature Scaling**: Compare different preprocessing methods

## 🧪 Educational Examples

### Example 1: Linear vs Non-linear Data

```python
from algorithms.linear_regression.linear_regression import LinearRegression
from algorithms.knn.knn import KNearestNeighbors

# Linear data - Linear Regression performs well
X_linear, y_linear = generate_linear_data(n_samples=100)
lr_model = LinearRegression()
lr_model.fit(X_linear, y_linear)

# Non-linear data - KNN performs better
X_nonlinear, y_nonlinear = generate_nonlinear_data(n_samples=100)
knn_model = KNearestNeighbors(k=5, task='regression')
knn_model.fit(X_nonlinear, y_nonlinear)
```

### Example 2: Classification Comparison

```python
# Compare algorithms on same dataset
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNearestNeighbors(k=5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"{name}: {accuracy:.3f}")
```

## 📊 Performance Characteristics

| Algorithm | Training Time | Prediction Time | Memory Usage | Interpretability |
|-----------|--------------|-----------------|--------------|-----------------|
| Linear Regression | O(n³) / O(kn) | O(1) | Low | High |
| Logistic Regression | O(kn) | O(1) | Low | High |
| KNN | O(1) | O(n) | High | Low |

*n = number of samples, k = number of iterations*

## 🔧 Customization and Extensions

### Adding New Algorithms

1. Create a new directory in `algorithms/`
2. Implement the algorithm following the existing pattern
3. Add comprehensive documentation
4. Include unit tests
5. Create examples

### Algorithm Template

```python
class NewAlgorithm:
    def __init__(self, hyperparameter=default_value):
        self.hyperparameter = hyperparameter
        # Initialize other attributes
    
    def fit(self, X, y):
        # Training logic
        return self
    
    def predict(self, X):
        # Prediction logic
        return predictions
    
    def score(self, X, y):
        # Evaluation logic
        return score
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Add new algorithms**: Implement additional classical ML algorithms
2. **Improve documentation**: Add more examples and explanations
3. **Optimize code**: Improve performance while maintaining clarity
4. **Add visualizations**: Create new plotting functions
5. **Fix bugs**: Report and fix any issues you find

## 📚 Further Reading

### Books
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Hands-On Machine Learning" by Aurélien Géron

### Online Resources
- [scikit-learn documentation](https://scikit-learn.org/)
- [Machine Learning Course by Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Linear Algebra Review](https://www.khanacademy.org/math/linear-algebra)

## ❓ FAQ

**Q: Why implement algorithms from scratch instead of using scikit-learn?**
A: Implementing from scratch helps you understand the mathematical foundations and algorithmic details that are abstracted away in high-level libraries.

**Q: Are these implementations production-ready?**
A: These implementations are educational tools. For production use, prefer well-tested libraries like scikit-learn that are optimized for performance and handle edge cases.

**Q: How do these implementations compare to scikit-learn?**
A: Our implementations should give very similar results to scikit-learn. We include comparison examples to demonstrate this.

**Q: Can I use this for my homework/project?**
A: Yes, but make sure to cite appropriately and check your institution's policies on using external code.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by classic machine learning textbooks and courses
- Built with educational clarity in mind
- Thanks to the open-source community for the tools and libraries that make this possible
