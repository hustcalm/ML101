# Tests

This directory contains the comprehensive test suite for the ML101 project. The tests ensure that all machine learning algorithm implementations work correctly, maintain consistent APIs, and produce expected results.

## Test Structure

### `test_algorithms.py`
Main test file containing unit tests for all algorithm implementations.

**Test Coverage:**
- ✅ Linear Regression (Normal Equation & Gradient Descent)
- ✅ Logistic Regression (Binary & Multiclass)
- ✅ K-Nearest Neighbors (Classification & Regression)
- ✅ Decision Trees (Classification & Regression)
- ✅ Naive Bayes (Gaussian, Multinomial, Bernoulli)
- ✅ K-Means Clustering
- ✅ Support Vector Machines (Multiple Kernels)
- ✅ Principal Component Analysis (PCA)
- ✅ Random Forest (Ensemble)
- ✅ Ridge Regression (L2 Regularization)
- ✅ Lasso Regression (L1 Regularization)
- ✅ Utility Functions (Metrics & Preprocessing)

## Running Tests

### Prerequisites
Install testing dependencies:
```bash
pip install pytest pytest-cov
```

### Run All Tests
```bash
# From the ML101 root directory
python -m pytest tests/
```

### Run Specific Test Classes
```bash
# Test only linear regression
python -m pytest tests/test_algorithms.py::TestLinearRegression

# Test only classification algorithms
python -m pytest tests/test_algorithms.py::TestLogisticRegression
python -m pytest tests/test_algorithms.py::TestKNearestNeighbors
python -m pytest tests/test_algorithms.py::TestDecisionTree
```

### Run with Coverage Report
```bash
# Generate coverage report
python -m pytest tests/ --cov=algorithms --cov=utils --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

### Verbose Output
```bash
# See detailed test output
python -m pytest tests/ -v

# See print statements and detailed failures
python -m pytest tests/ -v -s
```

## Test Categories

### Unit Tests
Each algorithm has comprehensive unit tests covering:

#### Initialization Tests
- Default parameter validation
- Custom parameter handling
- Invalid parameter rejection

#### Fitting Tests
- Model training on various datasets
- Parameter learning verification
- Edge case handling (empty data, single sample, etc.)

#### Prediction Tests
- Prediction accuracy on known datasets
- Output format consistency
- Prediction on new data points

#### API Consistency Tests
- Standard scikit-learn-like interface
- Method availability (fit, predict, score)
- Return type consistency

### Integration Tests
Tests that verify components work together:

#### Data Pipeline Tests
- Preprocessing → Algorithm → Evaluation
- Feature scaling integration
- Train/test split compatibility

#### Cross-Algorithm Tests
- Consistent results on same datasets
- Performance comparison validation
- Interface compatibility

### Performance Tests
Ensure algorithms meet performance expectations:

#### Accuracy Tests
- Minimum accuracy thresholds on known datasets
- Comparison with theoretical expectations
- Regression test for performance degradation

#### Speed Tests
- Maximum execution time limits
- Scalability with data size
- Memory usage constraints

## Test Data

### Synthetic Datasets
Most tests use synthetic data generators for consistent, reproducible testing:

```python
# Classification data
X, y = generate_classification_data(n_samples=100, n_features=5, n_classes=2)

# Regression data
X, y = generate_regression_data(n_samples=100, n_features=3, noise=0.1)

# Clustering data
X = generate_clustering_data(n_samples=150, n_centers=3, n_features=2)
```

### Real Datasets
Some tests use small real datasets for validation:
- Iris dataset (classification)
- Boston housing (regression)
- Wine dataset (multiclass classification)

## Test Examples

### Basic Algorithm Test
```python
def test_linear_regression_normal_equation(self):
    """Test linear regression with normal equation."""
    model = LinearRegression(method='normal')
    model.fit(self.X, self.y)
    
    # Check that model was fitted
    assert model.weights is not None
    assert model.bias is not None
    
    # Check predictions
    predictions = model.predict(self.X)
    assert len(predictions) == len(self.y)
    
    # Check R² score is reasonable
    r2 = model.score(self.X, self.y)
    assert r2 > 0.5  # Should have decent fit on synthetic data
```

### Parameter Validation Test
```python
def test_invalid_parameters(self):
    """Test that invalid parameters raise appropriate errors."""
    
    # Test invalid learning rate
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=-0.1)
    
    # Test invalid max_iterations
    with pytest.raises(ValueError):
        LogisticRegression(max_iterations=0)
    
    # Test invalid regularization strength
    with pytest.raises(ValueError):
        RidgeRegression(alpha=-1.0)
```

### Edge Case Test
```python
def test_single_sample(self):
    """Test behavior with single training sample."""
    X = np.array([[1, 2]])
    y = np.array([1])
    
    model = KNearestNeighbors(k=1)
    model.fit(X, y)
    
    prediction = model.predict(X)
    assert prediction[0] == y[0]
```

### Performance Comparison Test
```python
def test_performance_comparison(self):
    """Test that our implementation performs reasonably compared to baseline."""
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression
    
    # Train both models
    our_model = LinearRegression()
    sklearn_model = SklearnLinearRegression()
    
    our_model.fit(self.X, self.y)
    sklearn_model.fit(self.X, self.y)
    
    # Compare R² scores (should be very close)
    our_score = our_model.score(self.X, self.y)
    sklearn_score = sklearn_model.score(self.X, self.y)
    
    assert abs(our_score - sklearn_score) < 0.01
```

## Test Configuration

### pytest Configuration
Tests are configured via `pytest.ini` (if needed):

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

### Test Fixtures
Common test data and setup:

```python
@pytest.fixture
def classification_data():
    """Generate standard classification dataset for testing."""
    np.random.seed(42)
    return generate_classification_data(n_samples=100, n_features=5, n_classes=2)

@pytest.fixture
def regression_data():
    """Generate standard regression dataset for testing."""
    np.random.seed(42)
    return generate_regression_data(n_samples=100, n_features=3, noise=0.1)
```

## Continuous Integration

### GitHub Actions (if applicable)
Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled daily runs

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=algorithms --cov=utils
```

## Test-Driven Development

### Writing New Tests
When adding new algorithms:

1. **Write tests first**: Define expected behavior
2. **Start with simple cases**: Basic functionality
3. **Add edge cases**: Error conditions, boundary values
4. **Test integration**: How it works with utilities
5. **Performance tests**: Ensure reasonable speed/accuracy

### Test Structure Template
```python
class TestNewAlgorithm:
    """Test cases for NewAlgorithm implementation."""
    
    def setup_method(self):
        """Set up test data."""
        self.X, self.y = generate_test_data()
    
    def test_initialization(self):
        """Test algorithm initialization."""
        pass
    
    def test_fit(self):
        """Test model fitting."""
        pass
    
    def test_predict(self):
        """Test predictions."""
        pass
    
    def test_score(self):
        """Test scoring method."""
        pass
    
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        pass
    
    def test_edge_cases(self):
        """Test edge cases."""
        pass
```

## Test Maintenance

### Regular Tasks
- Update tests when algorithms change
- Add tests for new features
- Remove obsolete tests
- Update test data if needed
- Review and improve test coverage

### Code Quality
Tests should be:
- **Clear**: Easy to understand what's being tested
- **Isolated**: Each test independent of others
- **Fast**: Run quickly for frequent execution
- **Reliable**: Consistent results across runs
- **Comprehensive**: Cover all important functionality

## Debugging Failed Tests

### Common Issues
1. **Random seed differences**: Use fixed random seeds
2. **Floating point precision**: Use appropriate tolerances
3. **Platform differences**: Account for OS/hardware variations
4. **Dependency versions**: Pin important dependency versions

### Debugging Commands
```bash
# Run single failing test with maximum detail
python -m pytest tests/test_algorithms.py::TestLinearRegression::test_normal_equation -v -s --tb=long

# Run tests with pdb debugger on failure
python -m pytest tests/ --pdb

# Run tests with print statements visible
python -m pytest tests/ -s
```

## Coverage Goals

### Target Coverage
- **Algorithms**: >95% line coverage
- **Utilities**: >90% line coverage
- **Critical paths**: 100% coverage
- **Error handling**: >80% coverage

### Coverage Commands
```bash
# Generate detailed coverage report
python -m pytest tests/ --cov=algorithms --cov=utils --cov-report=term-missing

# Generate HTML coverage report
python -m pytest tests/ --cov=algorithms --cov=utils --cov-report=html

# Check coverage thresholds
python -m pytest tests/ --cov=algorithms --cov=utils --cov-fail-under=90
```

This comprehensive test suite ensures the reliability, correctness, and maintainability of all ML101 implementations, providing confidence in the educational and practical value of the codebase.
