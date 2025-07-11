# Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate predictor. It uses bootstrap aggregating (bagging) and random feature selection to reduce overfitting and improve generalization performance.

## Mathematical Foundation

### Ensemble Learning Principle

Random Forest combines the predictions of multiple decision trees:

**Classification (Majority Vote):**
```
ŷ = mode(T₁(x), T₂(x), ..., Tₙ(x))
```

**Regression (Average):**
```
ŷ = (1/n) * Σᵢ₌₁ⁿ Tᵢ(x)
```

Where Tᵢ(x) is the prediction of the i-th tree.

### Bootstrap Aggregating (Bagging)

Each tree is trained on a bootstrap sample of the original data:

1. **Bootstrap Sampling**: Sample n points with replacement from training set
2. **Train Tree**: Fit decision tree on bootstrap sample
3. **Repeat**: Create multiple trees with different bootstrap samples
4. **Aggregate**: Combine predictions from all trees

### Random Feature Selection

At each split in each tree:
1. Randomly select a subset of features (typically √p for classification, p/3 for regression)
2. Find the best split among these features only
3. This adds additional randomness and reduces correlation between trees

## Implementation Details

### Algorithm Steps

1. **For each tree in the forest:**
   - Create bootstrap sample of training data
   - Train decision tree with random feature selection
   - Store the trained tree

2. **For prediction:**
   - Get prediction from each tree
   - Combine predictions (vote/average)

3. **Calculate feature importances:**
   - Average feature importances across all trees

### Out-of-Bag (OOB) Error

- Each tree is trained on ~63% of data (bootstrap sample)
- Remaining ~37% are "out-of-bag" samples
- Use OOB samples to estimate model performance without separate validation set

### Hyperparameters

- **n_estimators**: Number of trees (more trees = better performance, slower training)
- **max_features**: Number of features per split (√p for classification, p/3 for regression)
- **max_depth**: Maximum depth of trees (controls overfitting)
- **min_samples_split**: Minimum samples to split a node
- **min_samples_leaf**: Minimum samples in leaf node

## Usage Example

```python
from random_forest import RandomForest, generate_sample_data

# Generate sample data
X, y = generate_sample_data(n_samples=1000, n_features=20, 
                           task='classification', random_state=42)

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Classification
rf_classifier = RandomForest(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    task='classification'
)

rf_classifier.fit(X_train, y_train)

# Predictions
predictions = rf_classifier.predict(X_test)
probabilities = rf_classifier.predict_proba(X_test)

# Evaluation
accuracy = rf_classifier.score(X_test, y_test)
oob_score = rf_classifier.oob_score_

print(f"Test Accuracy: {accuracy:.4f}")
print(f"OOB Score: {oob_score:.4f}")

# Feature importance
rf_classifier.plot_feature_importances()

# Regression example
rf_regressor = RandomForest(
    n_estimators=100,
    max_depth=10,
    task='regression',
    random_state=42
)

X_reg, y_reg = generate_sample_data(task='regression', random_state=42)
rf_regressor.fit(X_reg, y_reg)
r2_score = rf_regressor.score(X_reg, y_reg)
print(f"R² Score: {r2_score:.4f}")
```

## Parameters

- **n_estimators**: Number of trees in the forest
- **max_depth**: Maximum depth of trees
- **min_samples_split**: Minimum samples to split internal node
- **min_samples_leaf**: Minimum samples in leaf node
- **max_features**: Number of features per split ('sqrt', 'log2', int, float)
- **bootstrap**: Whether to use bootstrap sampling
- **oob_score**: Whether to compute out-of-bag score
- **random_state**: Random seed for reproducibility
- **task**: 'classification' or 'regression'

## Advantages

1. **Reduced Overfitting**: Ensemble reduces variance
2. **Robust to Outliers**: Individual trees may be affected, but ensemble is robust
3. **Feature Importance**: Provides feature importance scores
4. **Handles Missing Values**: Can handle missing values in features
5. **No Feature Scaling**: Tree-based methods don't require feature scaling
6. **Parallelizable**: Trees can be trained independently
7. **OOB Error**: Built-in validation without separate test set

## Disadvantages

1. **Less Interpretable**: Harder to interpret than single tree
2. **Memory Usage**: Stores multiple trees
3. **Slower Prediction**: Must query multiple trees
4. **Bias in Feature Selection**: Prefers features with more levels
5. **Overfitting with Noise**: Can still overfit with very noisy data

## Key Concepts

### Bootstrap Sampling
- Sample with replacement from training set
- Each bootstrap sample is same size as original
- On average, each sample contains ~63% of unique training instances

### Random Feature Selection
- At each split, randomly select subset of features
- Common choices:
  - Classification: √(number of features)
  - Regression: (number of features) / 3
  - Can be tuned as hyperparameter

### Feature Importance
Calculated as the average decrease in impurity when a feature is used for splitting across all trees:
```
importance(feature) = Σ (decrease in impurity) / n_trees
```

### Out-of-Bag Score
- Unbiased estimate of model performance
- Each tree makes predictions on data not used in its training
- Aggregate OOB predictions to get overall performance estimate

## Comparison with Single Decision Tree

| Aspect | Decision Tree | Random Forest |
|--------|---------------|---------------|
| Overfitting | High risk | Lower risk |
| Interpretability | High | Lower |
| Accuracy | Moderate | Higher |
| Training Speed | Fast | Slower |
| Prediction Speed | Fast | Slower |
| Memory Usage | Low | Higher |

## Hyperparameter Tuning

### Number of Estimators (n_estimators)
- More trees generally better, but diminishing returns
- Start with 100, increase if performance improves
- Balance between performance and computational cost

### Maximum Features (max_features)
- Too many: trees become similar (high correlation)
- Too few: trees become weak (high bias)
- Good defaults: √p (classification), p/3 (regression)

### Tree Depth (max_depth)
- Deeper trees: more complex, higher variance
- Shallower trees: simpler, higher bias
- Often no need to limit if using enough trees

### Example Tuning Strategy
```python
# Grid search example
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Use cross-validation to find best parameters
```

## Applications

### 1. Classification
- **Image Classification**: Object recognition, medical diagnosis
- **Text Classification**: Spam detection, sentiment analysis
- **Bioinformatics**: Gene expression analysis, protein classification

### 2. Regression
- **House Price Prediction**: Real estate valuation
- **Financial Forecasting**: Stock price prediction, risk assessment
- **Environmental Modeling**: Climate prediction, pollution modeling

### 3. Feature Selection
- Use feature importance scores to identify relevant features
- Remove low-importance features to reduce dimensionality

## Practical Tips

### 1. Start with Default Parameters
```python
rf = RandomForest(n_estimators=100, random_state=42)
```

### 2. Tune Number of Trees
```python
# Try different numbers of trees
for n in [50, 100, 200, 500]:
    rf = RandomForest(n_estimators=n)
    # Evaluate performance
```

### 3. Use OOB Score for Validation
```python
rf = RandomForest(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_}")
```

### 4. Feature Importance Analysis
```python
# Get feature importance
importances = rf.feature_importances_
# Sort features by importance
indices = np.argsort(importances)[::-1]
# Plot top features
rf.plot_feature_importances(feature_names=feature_names)
```

## Variants and Extensions

### 1. Extremely Randomized Trees (Extra Trees)
- Use random thresholds instead of optimal thresholds
- Even more randomness, faster training

### 2. Balanced Random Forest
- Handle class imbalance by balancing bootstrap samples
- Sample equal numbers from each class

### 3. Random Forest with Feature Selection
- Use recursive feature elimination with Random Forest
- Iteratively remove least important features

## Mathematical Intuition

Think of Random Forest as asking multiple experts (trees) for their opinion:
- Each expert has seen slightly different data (bootstrap)
- Each expert considers different aspects (random features)
- The final decision combines all expert opinions
- More experts generally lead to better decisions (up to a point)

The key insight is that averaging multiple models reduces variance while maintaining low bias, leading to better generalization.
