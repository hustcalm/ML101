# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of ML101 package
- Complete implementation of classical machine learning algorithms

## [0.1.0] - 2025-07-08

### Added
- **Linear Models**
  - Linear Regression (Normal Equation & Gradient Descent)
  - Logistic Regression with regularization
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)

- **Tree-Based Models**
  - Decision Trees (Classification & Regression)
  - Random Forest ensemble method

- **Instance-Based Learning**
  - K-Nearest Neighbors (KNN) classifier

- **Naive Bayes**
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes

- **Support Vector Machines**
  - SVM with linear and RBF kernels

- **Clustering**
  - K-Means clustering algorithm

- **Dimensionality Reduction**
  - Principal Component Analysis (PCA)

- **Utilities**
  - Standard Scaler for feature normalization
  - MinMax Scaler for feature scaling
  - Train-test split functionality
  - Comprehensive metrics for evaluation

### Documentation
- Comprehensive README with installation and usage instructions
- Algorithm-specific documentation in `docs/algorithms/`
- Jupyter notebooks with tutorials and examples
- API documentation with type hints

### Testing
- Unit tests for all algorithms
- Performance benchmarks against scikit-learn
- Continuous integration with GitHub Actions

### Examples
- Linear regression demonstrations
- Classification algorithm comparisons
- Advanced algorithms showcase
- Interactive Jupyter notebooks

### Infrastructure
- Professional package structure
- PyPI publishing configuration
- GitHub Actions CI/CD pipeline
- Code formatting and linting setup

---

## Version History

- **0.1.0** - Initial release with core algorithms
- **Unreleased** - Future improvements and additional algorithms

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information about contributing to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
