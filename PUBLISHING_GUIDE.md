# Publishing ML101 to PyPI

## Prerequisites

1. **Create PyPI account**: Register at https://pypi.org and https://test.pypi.org
2. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Step-by-Step Publishing Process

### 1. Prepare the Package

```bash
# Activate virtual environment
source venv/bin/activate

# Install package in development mode
pip install -e .

# Run tests to ensure everything works
python simple_demo.py
```

### 2. Build the Package

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
python -m build
```

This creates:
- `dist/ml101_algorithms-0.1.0.tar.gz` (source distribution)
- `dist/ml101_algorithms-0.1.0-py3-none-any.whl` (wheel distribution)

### 3. Test on TestPyPI First

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI to test
pip install --index-url https://test.pypi.org/simple/ ml101-algorithms
```

### 4. Upload to PyPI

```bash
# Upload to PyPI
twine upload dist/*
```

### 5. Verify Installation

```bash
# Install from PyPI
pip install ml101-algorithms

# Test import
python -c "import ml101; print(ml101.__version__)"
```

## Package Usage Examples

After publishing, users can install and use your package:

```python
# Install
pip install ml101-algorithms

# Use
from ml101 import LinearRegression, KNearestNeighbors
from ml101.utils import train_test_split, StandardScaler

# Create and use models
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Current Package Status

✅ **Package Structure**: Complete  
✅ **Core Algorithms**: 9 algorithms implemented  
✅ **Documentation**: Comprehensive READMEs  
✅ **Setup Files**: setup.py, pyproject.toml, MANIFEST.in  
✅ **Testing**: Basic functionality verified  

## Ready for Publishing!

Your ML101 package is now ready to be published to PyPI. The package includes:

- **Linear Models**: LinearRegression, LogisticRegression, RidgeRegression, LassoRegression
- **Tree Models**: DecisionTree, RandomForest  
- **Clustering**: KMeans
- **Neighbors**: KNearestNeighbors
- **Naive Bayes**: GaussianNaiveBayes
- **Decomposition**: PCA
- **Utilities**: StandardScaler, MinMaxScaler, train_test_split

## Version Management

Update version numbers in:
- `setup.py`
- `ml101/__init__.py`
- `pyproject.toml`

## Next Steps

1. Test the build process
2. Upload to TestPyPI
3. Verify installation works
4. Upload to PyPI
5. Share with the community!
