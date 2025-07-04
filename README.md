# ML101: Classical Machine Learning Algorithms

A comprehensive implementation of classical machine learning algorithms from scratch, with detailed explanations and examples.

## 🎯 Project Overview

This project implements classical machine learning algorithms from the ground up, providing:
- Clean, well-documented implementations
- Comprehensive examples and visualizations
- Theoretical explanations
- Performance comparisons with scikit-learn

## 📚 Algorithms Covered

### Supervised Learning

#### Linear Models
- [Linear Regression](./algorithms/linear_regression/) ✅
- [Logistic Regression](./algorithms/logistic_regression/) ✅
- [Ridge Regression](./algorithms/ridge_regression/) ✅
- [Lasso Regression](./algorithms/lasso_regression/) ✅

#### Tree-Based Models
- [Decision Trees](./algorithms/decision_trees/) ✅
- [Random Forest](./algorithms/random_forest/) ✅

#### Instance-Based Learning
- [K-Nearest Neighbors (KNN)](./algorithms/knn/) ✅

#### Support Vector Machines
- [SVM](./algorithms/svm/) ✅

#### Naive Bayes
- [Gaussian Naive Bayes](./algorithms/naive_bayes/) ✅

#### Ensemble Methods
- [Random Forest](./algorithms/random_forest/) ✅
- [AdaBoost](./algorithms/adaboost/) 🚧
- [Gradient Boosting](./algorithms/gradient_boosting/) 🚧

### Unsupervised Learning

#### Clustering
- [K-Means](./algorithms/kmeans/) ✅
- [Hierarchical Clustering](./algorithms/hierarchical_clustering/) 🚧
- [DBSCAN](./algorithms/dbscan/) 🚧

#### Dimensionality Reduction
- [Principal Component Analysis (PCA)](./algorithms/pca/) ✅
- [Linear Discriminant Analysis (LDA)](./algorithms/lda/) 🚧

### Model Selection & Evaluation
- [Cross Validation](./utils/cross_validation.py)
- [Metrics](./utils/metrics.py)
- [Preprocessing](./utils/preprocessing.py)

## 🚀 Quick Start

1. **Clone and navigate to the project:**
   ```bash
   cd /home/lihli/Repos/ML101
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # On Linux/Mac
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run examples:**
   ```bash
   # Linear Regression example
   python examples/linear_regression_example.py
   
   # Classification comparison
   python examples/classification_comparison.py
   
   # Advanced algorithms (SVM, PCA, Random Forest)
   python examples/advanced_algorithms_demo.py
   
   # Clustering example
   python examples/clustering_example.py
   ```

5. **Run interactive notebooks:**
   ```bash
   jupyter lab notebooks/
   ```

6. **Deactivate virtual environment when done:**
   ```bash
   deactivate
   ```

## 📁 Project Structure

```
ML101/
├── algorithms/                 # Algorithm implementations
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── decision_trees/
│   └── ...
├── examples/                   # Runnable examples
├── notebooks/                  # Jupyter notebooks
├── utils/                      # Utility functions
├── datasets/                   # Sample datasets
├── tests/                      # Unit tests
├── docs/                       # Documentation
└── requirements.txt
```

## 🧪 Running Tests

```bash
python -m pytest tests/
```

## 📖 Documentation

Detailed documentation for each algorithm can be found in the `docs/` directory or within each algorithm's folder.

## 🤝 Contributing

Feel free to contribute by adding new algorithms, improving documentation, or fixing bugs!

## 📄 License

MIT License - see LICENSE file for details.
