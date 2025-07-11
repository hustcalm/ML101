# ML101 Algorithm Documentation

This directory contains detailed documentation for all machine learning algorithms implemented in the ML101 project. Each algorithm has its own comprehensive README file with mathematical foundations, implementation details, usage examples, and best practices.

## 📚 Available Algorithms

### Supervised Learning

#### Linear Models
- **[Linear Regression](./linear_regression_README.md)** - Fundamental regression algorithm with normal equation and gradient descent methods
- **[Logistic Regression](./logistic_regression_README.md)** - Binary and multiclass classification with regularization options
- **[Ridge Regression](./ridge_regression_README.md)** - L2 regularized regression for handling multicollinearity and overfitting
- **[Lasso Regression](./lasso_regression_README.md)** - L1 regularized regression with automatic feature selection

#### Tree-Based Models
- **[Decision Trees](./decision_trees_README.md)** - Interpretable classification and regression trees with various splitting criteria
- **[Random Forest](./random_forest_README.md)** - Ensemble method combining multiple decision trees with bootstrap aggregating

#### Instance-Based Learning
- **[K-Nearest Neighbors](./knn_README.md)** - Non-parametric algorithm for classification and regression

#### Probabilistic Models
- **[Naive Bayes](./naive_bayes_README.md)** - Probabilistic classifier based on Bayes' theorem with feature independence assumption

#### Support Vector Machines
- **[SVM](./svm_README.md)** - Maximum margin classifier with kernel tricks for non-linear boundaries

### Unsupervised Learning

#### Clustering
- **[K-Means](./kmeans_README.md)** - Partitional clustering algorithm using centroid-based approach

#### Dimensionality Reduction
- **[Principal Component Analysis](./pca_README.md)** - Linear dimensionality reduction using eigenvalue decomposition

## 📖 Documentation Structure

Each algorithm documentation includes:

### 1. **Mathematical Foundation**
- Core equations and theoretical background
- Optimization objectives and methods
- Assumptions and properties

### 2. **Implementation Details**
- Algorithm steps and pseudocode
- Key parameters and hyperparameters
- Computational complexity analysis

### 3. **Usage Examples**
- Practical code examples
- Parameter tuning guidance
- Visualization techniques

### 4. **Advantages & Disadvantages**
- When to use each algorithm
- Strengths and limitations
- Comparison with alternatives

### 5. **Best Practices**
- Data preprocessing requirements
- Common pitfalls to avoid
- Performance optimization tips

## 🚀 Quick Reference

| Algorithm | Type | Key Strength | Main Use Case |
|-----------|------|--------------|---------------|
| Linear Regression | Regression | Interpretability | Simple linear relationships |
| Logistic Regression | Classification | Probability estimates | Binary/multiclass classification |
| Ridge Regression | Regression | Handles multicollinearity | Regularized linear regression |
| Lasso Regression | Regression | Feature selection | Sparse linear models |
| Decision Trees | Both | Interpretability | Non-linear relationships |
| Random Forest | Both | Robustness | General-purpose ensemble |
| K-Nearest Neighbors | Both | Simplicity | Non-parametric problems |
| Naive Bayes | Classification | Speed | Text classification |
| SVM | Classification | High accuracy | Complex decision boundaries |
| K-Means | Clustering | Efficiency | Spherical clusters |
| PCA | Dimensionality Reduction | Variance preservation | Data visualization |

## 🔗 Integration with Package

All algorithms are available through the ML101 package:

```python
from ml101 import (
    LinearRegression, LogisticRegression, RidgeRegression, LassoRegression,
    DecisionTree, RandomForest, KNearestNeighbors, 
    GaussianNaiveBayes, KMeans, PCA
)
from ml101.utils import StandardScaler, train_test_split
```

## 📝 Contributing

When adding new algorithm documentation:

1. **Follow the established structure** outlined above
2. **Include comprehensive examples** with real data
3. **Add mathematical derivations** where appropriate
4. **Provide visual illustrations** when helpful
5. **Cross-reference related algorithms** for comparison

## 🎯 Educational Goals

This documentation is designed to:
- **Teach fundamental ML concepts** through clear explanations
- **Bridge theory and practice** with working implementations
- **Provide hands-on learning** through executable examples
- **Develop intuition** about when to use each algorithm
- **Establish best practices** for ML workflows

## 📚 Additional Resources

For more comprehensive learning:
- **[Main Project Documentation](../README.md)** - Complete project overview
- **[Examples Directory](../../examples/)** - Working code examples
- **[Notebooks](../../notebooks/)** - Interactive tutorials
- **[Tests](../../tests/)** - Algorithm validation and examples

---

*This documentation is part of the ML101 project - a comprehensive educational resource for learning machine learning algorithms from scratch.*
