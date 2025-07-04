# Contributing to ML101

We welcome contributions to the ML101 project! This guide will help you get started with contributing to our machine learning algorithms implementation.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic understanding of machine learning concepts
- Familiarity with NumPy and mathematical concepts
- Git knowledge for version control

### Setting Up Development Environment

1. **Fork and clone the repository:**
```bash
git clone https://github.com/your-username/ML101.git
cd ML101
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing
```

4. **Verify setup:**
```bash
python -m pytest tests/
```

## Types of Contributions

### 1. Bug Fixes
- Fix implementation errors in existing algorithms
- Correct mathematical formulations
- Improve error handling and edge cases
- Fix documentation typos

### 2. New Algorithm Implementations
- Add new classical ML algorithms
- Implement algorithm variants
- Add new utility functions

### 3. Documentation Improvements
- Enhance algorithm explanations
- Add mathematical derivations
- Improve code comments
- Create or update examples

### 4. Testing
- Add unit tests for existing algorithms
- Improve test coverage
- Add integration tests
- Performance testing

### 5. Examples and Tutorials
- Create new example scripts
- Add Jupyter notebooks
- Improve existing examples
- Add visualization examples

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/algorithm-name
# or
git checkout -b fix/issue-description
```

### 2. Implementation Guidelines

#### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add comprehensive docstrings
- Include type hints where appropriate

#### Algorithm Implementation Structure
```python
"""
Algorithm Name Implementation

Brief description of the algorithm and its purpose.

Mathematical Foundation:
- Key equations and formulations
- Optimization methods used
- Time/space complexity

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple

class AlgorithmName:
    """
    Brief description of the algorithm.
    
    Parameters:
    -----------
    param1 : type
        Description of parameter 1
    param2 : type, default=value
        Description of parameter 2
    
    Attributes:
    -----------
    attribute1_ : type
        Description of fitted attribute
    """
    
    def __init__(self, param1: type, param2: type = default_value):
        self.param1 = param1
        self.param2 = param2
        
        # Initialize attributes that will be set during fitting
        self.attribute1_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AlgorithmName':
        """
        Fit the algorithm to training data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training targets
            
        Returns:
        --------
        self : AlgorithmName
            Returns self for method chaining
        """
        # Implementation here
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values
        """
        # Implementation here
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate performance score.
        
        Parameters:
        -----------
        X : np.ndarray
            Test features
        y : np.ndarray
            Test targets
            
        Returns:
        --------
        score : float
            Performance score (accuracy for classification, R² for regression)
        """
        # Implementation here
        pass

def generate_sample_data(n_samples: int = 100, 
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for algorithm demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    """
    # Implementation here
    pass

# Example usage and demonstration
if __name__ == "__main__":
    print("Algorithm Name Implementation Demo")
    print("=" * 40)
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=100, random_state=42)
    
    # Train algorithm
    algorithm = AlgorithmName(param1=value1, param2=value2)
    algorithm.fit(X, y)
    
    # Make predictions and evaluate
    predictions = algorithm.predict(X)
    score = algorithm.score(X, y)
    
    print(f"Performance score: {score:.4f}")
    
    print("Demo completed!")
```

#### Directory Structure for New Algorithms
```
algorithms/
└── algorithm_name/
    ├── algorithm_name.py          # Main implementation
    ├── README.md                  # Documentation
    └── __pycache__/              # Generated during testing
```

### 3. Documentation Requirements

#### README.md Template
```markdown
# Algorithm Name

Brief description of what the algorithm does and its use cases.

## Mathematical Foundation

### Problem Formulation
Describe the problem the algorithm solves.

### Algorithm Description
Explain the key steps and mathematical formulations.

### Optimization Method
Describe how the algorithm finds the optimal solution.

## Implementation Details

### Key Features
- Feature 1
- Feature 2
- Feature 3

### Parameters
- **param1** (type): Description
- **param2** (type, default): Description

## Usage Example

\```python
from algorithm_name import AlgorithmName

# Create and train model
model = AlgorithmName(param1=value1)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
\```

## Advantages and Disadvantages

### Advantages
- Advantage 1
- Advantage 2

### Disadvantages
- Disadvantage 1
- Disadvantage 2

## When to Use

Describe scenarios where this algorithm is appropriate.

## Comparison with Other Methods

Compare with similar algorithms in the project.
```

### 4. Testing Requirements

Every new algorithm must include comprehensive tests:

```python
class TestAlgorithmName:
    """Test cases for AlgorithmName implementation."""
    
    def setup_method(self):
        """Set up test data."""
        self.X, self.y = generate_sample_data(n_samples=100, random_state=42)
    
    def test_initialization(self):
        """Test algorithm initialization."""
        algorithm = AlgorithmName(param1=value1)
        assert algorithm.param1 == value1
    
    def test_fit(self):
        """Test model fitting."""
        algorithm = AlgorithmName()
        algorithm.fit(self.X, self.y)
        assert algorithm.attribute1_ is not None
    
    def test_predict(self):
        """Test predictions."""
        algorithm = AlgorithmName()
        algorithm.fit(self.X, self.y)
        predictions = algorithm.predict(self.X)
        assert len(predictions) == len(self.y)
    
    def test_score(self):
        """Test scoring method."""
        algorithm = AlgorithmName()
        algorithm.fit(self.X, self.y)
        score = algorithm.score(self.X, self.y)
        assert 0 <= score <= 1  # Adjust based on algorithm
    
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            AlgorithmName(param1=invalid_value)
```

### 5. Example Script

Create an example script in the `examples/` directory:

```python
"""
Algorithm Name Example

Demonstrates the usage of AlgorithmName implementation.
"""

def main():
    # Implementation here
    pass

if __name__ == "__main__":
    main()
```

## Submission Process

### 1. Before Submitting
- [ ] Code follows style guidelines
- [ ] All tests pass (`python -m pytest tests/`)
- [ ] Documentation is complete and accurate
- [ ] Example script works correctly
- [ ] Performance is reasonable compared to existing implementations

### 2. Pull Request Process

1. **Push your branch:**
```bash
git add .
git commit -m "Add [algorithm name] implementation"
git push origin feature/algorithm-name
```

2. **Create Pull Request:**
- Use descriptive title: "Add [Algorithm Name] implementation"
- Fill out the PR template
- Include summary of changes
- Reference any related issues

3. **PR Description Template:**
```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New algorithm implementation
- [ ] Documentation improvement
- [ ] Test improvement
- [ ] Example addition

## Algorithm Details (for new algorithms)
- **Algorithm**: Name of the algorithm
- **Type**: Classification/Regression/Clustering/etc.
- **Complexity**: Time and space complexity
- **Key Features**: Notable features or capabilities

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Example script runs without errors

## Documentation
- [ ] Algorithm README.md created/updated
- [ ] Docstrings are comprehensive
- [ ] Mathematical formulations included
- [ ] Usage examples provided

## Checklist
- [ ] Code follows PEP 8 style guidelines
- [ ] No unnecessary dependencies added
- [ ] Performance is reasonable
- [ ] Integration with existing utilities works
```

### 3. Review Process
- Maintainers will review your code
- Address any feedback or requested changes
- Once approved, your contribution will be merged

## Code Review Guidelines

### For Reviewers
- Check mathematical correctness
- Verify API consistency
- Ensure comprehensive testing
- Review documentation quality
- Test performance and accuracy

### For Contributors
- Be responsive to feedback
- Ask questions if unclear about requests
- Make requested changes promptly
- Test thoroughly before resubmitting

## Best Practices

### Code Quality
1. **Single Responsibility**: Each class/function should have one clear purpose
2. **Consistent Naming**: Use clear, descriptive names
3. **Error Handling**: Include appropriate error checking and messages
4. **Performance**: Optimize for readability first, then performance
5. **Documentation**: Write code that documents itself

### Mathematical Accuracy
1. **Verify Formulations**: Double-check mathematical equations
2. **Numerical Stability**: Consider numerical precision issues
3. **Edge Cases**: Handle boundary conditions appropriately
4. **Convergence**: Ensure iterative algorithms converge properly

### Educational Value
1. **Clear Explanations**: Make complex concepts understandable
2. **Step-by-Step**: Break down complex operations
3. **Visualizations**: Include helpful plots where appropriate
4. **Examples**: Provide practical usage examples

## Getting Help

### Resources
- Check existing implementations for patterns
- Read algorithm documentation in the `docs/` directory
- Look at test files for expected behavior
- Review example scripts for usage patterns

### Communication
- Open an issue for questions or discussions
- Join our community discussions
- Ask for help in your pull request comments

### Common Issues
1. **Import Errors**: Check Python path and module structure
2. **Test Failures**: Ensure all dependencies are installed
3. **Performance Issues**: Profile code and optimize bottlenecks
4. **Documentation**: Use clear, concise language with examples

## Recognition

Contributors will be:
- Listed in the project acknowledgments
- Credited in algorithm documentation
- Recognized in release notes for significant contributions

Thank you for contributing to ML101! Your efforts help make machine learning education more accessible to everyone.
