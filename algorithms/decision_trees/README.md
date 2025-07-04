# Decision Trees

## Overview

Decision Trees are versatile machine learning algorithms that can be used for both classification and regression tasks. They work by recursively splitting the data based on feature values to create a tree-like structure that makes predictions.

## Mathematical Foundation

### Decision Tree Algorithm

1. **Start with the root node** containing all training samples
2. **For each node**, find the best feature and threshold to split the data
3. **Create child nodes** based on the split
4. **Repeat recursively** until stopping criteria are met
5. **Assign predictions** to leaf nodes

### Splitting Criteria

#### For Classification:

**Gini Impurity:**
```
Gini(S) = 1 - Σ(pi²)
```
where pi is the proportion of samples belonging to class i.

**Entropy (Information Gain):**
```
Entropy(S) = -Σ(pi * log2(pi))
Information Gain = Entropy(parent) - Σ(|Si|/|S| * Entropy(Si))
```

#### For Regression:

**Mean Squared Error:**
```
MSE(S) = (1/n) * Σ(yi - ȳ)²
```

**Mean Absolute Error:**
```
MAE(S) = (1/n) * Σ|yi - median(y)|
```

### Stopping Criteria

- Maximum depth reached
- Minimum samples per split
- Minimum samples per leaf
- No improvement in impurity
- All samples have same target value

## Key Concepts

### Advantages

1. **Interpretability**: Easy to understand and visualize
2. **No assumptions**: About data distribution
3. **Handles mixed data**: Both numerical and categorical features
4. **Feature selection**: Automatically selects relevant features
5. **Non-linear relationships**: Can capture complex patterns

### Disadvantages

1. **Overfitting**: Prone to creating overly complex trees
2. **Instability**: Small changes in data can result in different trees
3. **Bias**: Can be biased toward features with more levels
4. **Limited expressiveness**: Each split is axis-aligned

### Hyperparameters

- **max_depth**: Maximum depth of the tree
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required at a leaf node
- **criterion**: Splitting criterion (gini, entropy, mse, mae)
- **max_features**: Number of features to consider for each split

## Implementation Features

Our implementation includes:

- **Both classification and regression**
- **Multiple splitting criteria**
- **Pruning parameters** to prevent overfitting
- **Feature importance calculation**
- **Tree structure visualization**
- **Decision boundary plotting**

## Usage Example

```python
from decision_tree import DecisionTree, generate_tree_data

# Generate sample data
X, y = generate_tree_data(task='classification', n_samples=200, random_state=42)

# Create and train decision tree
tree = DecisionTree(
    task='classification',
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)

# Evaluate
accuracy = tree.score(X, y)
print(f"Accuracy: {accuracy:.4f}")

# Get tree information
print(f"Tree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")

# Feature importances
importances = tree.feature_importances_()
print(f"Feature importances: {importances}")
```

## Hyperparameter Tuning

```python
# Example of hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Note: This example uses sklearn's GridSearchCV for demonstration
# You could implement your own parameter search
```

## Preventing Overfitting

### Pre-pruning (Early Stopping)
- Set `max_depth` to limit tree depth
- Set `min_samples_split` to require minimum samples for splitting
- Set `min_samples_leaf` to require minimum samples at leaf nodes

### Post-pruning
- Build full tree first
- Remove nodes that don't improve validation performance
- Use techniques like Reduced Error Pruning or Cost Complexity Pruning

## When to Use Decision Trees

**Good for:**
- **Interpretable models** where you need to explain decisions
- **Mixed data types** (numerical and categorical)
- **Non-linear relationships** in the data
- **Feature selection** and understanding feature importance
- **Baseline models** for comparison

**Not ideal for:**
- **Linear relationships** (linear models work better)
- **Small datasets** (prone to overfitting)
- **Noisy data** (can memorize noise)
- **When stability is important** (small changes affect structure)

## Comparison with Other Algorithms

| Aspect | Decision Trees | Linear Models | KNN |
|--------|----------------|---------------|-----|
| Interpretability | High | High | Low |
| Non-linearity | Yes | No | Yes |
| Training Speed | Fast | Fast | Instant |
| Prediction Speed | Fast | Fast | Slow |
| Overfitting Risk | High | Low | Medium |
| Feature Scaling | Not needed | Needed | Needed |

## Tree Structure Visualization

```python
from decision_tree import DecisionTreeVisualization

# Print tree structure
DecisionTreeVisualization.plot_tree_structure(tree, feature_names=['X1', 'X2'])

# Plot decision boundary (for 2D data)
DecisionTreeVisualization.plot_decision_boundary(tree, X, y)

# Plot feature importances
DecisionTreeVisualization.plot_feature_importances(tree, feature_names=['X1', 'X2'])
```

## Real-World Applications

1. **Medical Diagnosis**: Decision trees for medical decision making
2. **Credit Scoring**: Loan approval decisions
3. **Customer Segmentation**: Marketing strategy decisions
4. **Fraud Detection**: Rule-based fraud identification
5. **Feature Engineering**: Understanding feature relationships

## Extensions

Decision Trees are building blocks for more advanced algorithms:

- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential tree building
- **XGBoost/LightGBM**: Optimized boosting implementations
- **Extra Trees**: Extremely randomized trees
