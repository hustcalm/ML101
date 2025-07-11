# Naive Bayes Classifier

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with the "naive" assumption of conditional independence between features. Despite this strong assumption, Naive Bayes classifiers often perform surprisingly well in practice, especially for text classification and spam filtering.

## Mathematical Foundation

### Bayes' Theorem

Naive Bayes is built on Bayes' theorem:

```
P(y|X) = P(X|y) × P(y) / P(X)
```

Where:
- P(y|X): Posterior probability of class y given features X
- P(X|y): Likelihood of features X given class y
- P(y): Prior probability of class y
- P(X): Evidence (marginal probability of features X)

### Naive Assumption

The "naive" assumption is that features are conditionally independent given the class:

```
P(X|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y) = ∏ᵢ₌₁ⁿ P(xᵢ|y)
```

### Classification Decision

For classification, we predict the class with maximum posterior probability:

```
ŷ = argmax P(y|X) = argmax P(y) × ∏ᵢ₌₁ⁿ P(xᵢ|y)
```

## Naive Bayes Variants

### 1. Gaussian Naive Bayes

For continuous features that follow a normal distribution:

```
P(xᵢ|y) = (1/√(2πσ²ᵧ)) × exp(-((xᵢ - μᵧ)²)/(2σ²ᵧ))
```

Where:
- μᵧ: Mean of feature xᵢ for class y
- σ²ᵧ: Variance of feature xᵢ for class y

### 2. Multinomial Naive Bayes

For discrete features (e.g., word counts):

```
P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α × n)
```

Where:
- count(xᵢ, y): Count of feature xᵢ in class y
- α: Smoothing parameter (Laplace smoothing)
- n: Number of features

### 3. Bernoulli Naive Bayes

For binary features:

```
P(xᵢ|y) = P(xᵢ = 1|y) × xᵢ + (1 - P(xᵢ = 1|y)) × (1 - xᵢ)
```

## Key Properties

### Conditional Independence

The naive assumption assumes that features are independent given the class label. While this is rarely true in practice, it often works well.

### Probabilistic Output

Naive Bayes naturally provides probability estimates for class membership, not just classifications.

### Handling Missing Values

Can easily handle missing values by simply ignoring them in the probability calculations.

## Advantages

1. **Simple and Fast**: Easy to implement and computationally efficient
2. **Small Training Set**: Performs well with limited training data
3. **Handles Multiple Classes**: Naturally handles multi-class classification
4. **Probabilistic Output**: Provides class probability estimates
5. **Insensitive to Irrelevant Features**: Irrelevant features don't significantly impact performance
6. **Good Baseline**: Often used as a baseline for comparison
7. **Handles Categorical and Continuous Features**: Different variants for different data types

## Disadvantages

1. **Strong Independence Assumption**: Assumes features are independent (rarely true)
2. **Categorical Inputs**: Continuous features require discretization for some variants
3. **Zero Probability Problem**: Can assign zero probability to unseen combinations
4. **Limited Expressiveness**: Cannot capture feature interactions
5. **Sensitive to Skewed Data**: Performance can degrade with highly skewed datasets
6. **Poor Estimator**: Probability estimates can be poor, though classifications are often good

## Implementation Features

Our implementation includes:

- **Gaussian Naive Bayes**: For continuous features with normal distribution
- **Multinomial Naive Bayes**: For discrete count features
- **Bernoulli Naive Bayes**: For binary features
- **Laplace Smoothing**: Handles zero probability problem
- **Probability Estimates**: Provides class probability predictions
- **Robust Implementation**: Handles numerical stability issues

## Usage Example

### Gaussian Naive Bayes

```python
from ml101 import GaussianNaiveBayes
import numpy as np

# Generate sample data
X_train = np.random.randn(100, 4)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

# Create and fit Gaussian Naive Bayes
gnb = GaussianNaiveBayes(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

# Make predictions
X_test = np.random.randn(20, 4)
predictions = gnb.predict(X_test)
print(f"Predictions: {predictions}")

# Get class probabilities
probabilities = gnb.predict_proba(X_test)
print(f"Class probabilities: {probabilities}")
```

### Multinomial Naive Bayes

```python
from ml101 import MultinomialNaiveBayes

# Document-term matrix (word counts)
X_train = np.array([
    [1, 2, 0, 1],  # Document 1: word counts
    [0, 1, 1, 0],  # Document 2: word counts
    [2, 0, 1, 1],  # Document 3: word counts
    # ... more documents
])
y_train = np.array([0, 1, 0])  # Class labels

# Create and fit Multinomial Naive Bayes
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_train, y_train)

# Make predictions
X_test = np.array([[1, 1, 0, 1]])
predictions = mnb.predict(X_test)
print(f"Predictions: {predictions}")
```

### Bernoulli Naive Bayes

```python
from ml101 import BernoulliNaiveBayes

# Binary features
X_train = np.array([
    [1, 1, 0, 1],  # Document 1: word presence
    [0, 1, 1, 0],  # Document 2: word presence
    [1, 0, 1, 1],  # Document 3: word presence
    # ... more documents
])
y_train = np.array([0, 1, 0])  # Class labels

# Create and fit Bernoulli Naive Bayes
bnb = BernoulliNaiveBayes(alpha=1.0)
bnb.fit(X_train, y_train)

# Make predictions
X_test = np.array([[1, 1, 0, 1]])
predictions = bnb.predict(X_test)
print(f"Predictions: {predictions}")
```

## Parameters

### Gaussian Naive Bayes
- **var_smoothing** (float, default=1e-9): Smoothing parameter for variance calculation

### Multinomial Naive Bayes
- **alpha** (float, default=1.0): Additive smoothing parameter
- **fit_prior** (bool, default=True): Whether to learn class prior probabilities

### Bernoulli Naive Bayes
- **alpha** (float, default=1.0): Additive smoothing parameter
- **binarize** (float, default=0.0): Threshold for binarizing features

## Attributes

- **classes_**: Array of class labels
- **class_priors_**: Prior probabilities for each class
- **feature_log_prob_**: Log probabilities of features given classes
- **class_count_**: Number of samples per class
- **feature_count_**: Number of samples per feature per class

## Handling the Zero Probability Problem

### Laplace Smoothing

Add a small constant (α) to all counts to avoid zero probabilities:

```python
# For Multinomial Naive Bayes
P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α × V)
```

Where V is the vocabulary size.

### Add-k Smoothing

Generalization of Laplace smoothing with parameter k:

```python
P(xᵢ|y) = (count(xᵢ, y) + k) / (count(y) + k × V)
```

## Model Evaluation

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Evaluate model performance
scores = cross_val_score(gnb, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, classification_report

# Get predictions
y_pred = gnb.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Applications

1. **Text Classification**: Spam detection, sentiment analysis, document categorization
2. **Medical Diagnosis**: Disease prediction based on symptoms
3. **Recommendation Systems**: Content-based filtering
4. **Real-time Predictions**: Fast classification in streaming applications
5. **Baseline Models**: Quick baseline for comparison with complex models
6. **Feature Selection**: Identify important features for classification
7. **Anomaly Detection**: Detect unusual patterns in data

## Comparison with Other Algorithms

| Algorithm | Advantages | Disadvantages |
|-----------|------------|---------------|
| Naive Bayes | Fast, simple, works with small data | Strong independence assumption |
| Logistic Regression | No independence assumption, interpretable | Requires more data, sensitive to outliers |
| Decision Trees | Can capture interactions, interpretable | Can overfit, biased toward certain features |
| SVM | Effective in high dimensions | Requires feature scaling, no probabilities |
| KNN | No assumptions, simple | Computationally expensive, sensitive to irrelevant features |

## Text Classification Example

```python
from sklearn.feature_extraction.text import CountVectorizer
from ml101 import MultinomialNaiveBayes

# Sample text data
texts = [
    "I love this movie",
    "This movie is terrible",
    "Great film with amazing acting",
    "Boring and poorly made",
    "Fantastic storyline and characters"
]
labels = [1, 0, 1, 0, 1]  # 1: positive, 0: negative

# Convert text to word counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Train Naive Bayes
nb = MultinomialNaiveBayes(alpha=1.0)
nb.fit(X, labels)

# Test on new text
test_text = ["This movie is amazing"]
test_X = vectorizer.transform(test_text).toarray()
prediction = nb.predict(test_X)
probability = nb.predict_proba(test_X)

print(f"Prediction: {prediction[0]}")
print(f"Probabilities: {probability[0]}")
```

## Feature Engineering for Naive Bayes

### Text Processing

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Use TF-IDF instead of raw counts
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf.fit_transform(texts)
```

### Handling Continuous Features

```python
from sklearn.preprocessing import KBinsDiscretizer

# Discretize continuous features for Multinomial NB
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal')
X_discrete = discretizer.fit_transform(X_continuous)
```

## Mathematical Complexity

- **Time Complexity**: 
  - Training: O(n × d)
  - Prediction: O(d × c)
  - n: number of samples
  - d: number of features
  - c: number of classes

- **Space Complexity**: O(d × c)
  - Store parameters for each feature-class combination

## Tips for Better Results

1. **Choose Right Variant**: Gaussian for continuous, Multinomial for counts, Bernoulli for binary
2. **Feature Engineering**: Create relevant features and remove irrelevant ones
3. **Handle Imbalanced Data**: Use appropriate metrics and sampling techniques
4. **Smooth Parameters**: Tune smoothing parameters to avoid overfitting
5. **Combine with Other Methods**: Use as part of ensemble methods
6. **Validate Assumptions**: Check if independence assumption is reasonable

## Common Pitfalls

1. **Ignoring Feature Correlations**: May perform poorly when features are highly correlated
2. **Inappropriate Variant**: Using wrong variant for data type
3. **No Feature Selection**: Including irrelevant features can hurt performance
4. **Ignoring Class Imbalance**: May be biased toward majority class
5. **Poor Probability Calibration**: Probability estimates may be poorly calibrated

## References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval
2. Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective
3. Rish, I. (2001). An empirical study of the naive Bayes classifier
4. Zhang, H. (2004). The optimality of naive Bayes