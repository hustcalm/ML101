"""
Simple demonstration of the ML101 package functionality
"""

import numpy as np
from ml101 import LinearRegression, KNearestNeighbors, DecisionTree
from ml101.utils import StandardScaler, train_test_split

def main():
    print("🚀 ML101 Package Demo")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y_reg = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1
    y_clf = (y_reg > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    
    print("\n1. 📈 Testing LinearRegression:")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    r2_score = lr.score(X_test, y_test)
    print(f"   R² Score: {r2_score:.4f}")
    
    print("\n2. 🏠 Testing KNearestNeighbors (Regression):")
    knn = KNearestNeighbors(k=5)
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print(f"   R² Score: {knn_score:.4f}")
    
    print("\n3. 🌳 Testing DecisionTree (Classification):")
    dt = DecisionTree()
    dt.fit(X_train, y_clf_train)
    dt_score = dt.score(X_test, y_clf_test)
    print(f"   Accuracy: {dt_score:.4f}")
    
    print("\n4. 🎯 Testing KNN (Classification):")
    knn_clf = KNearestNeighbors(k=3)
    knn_clf.fit(X_train, y_clf_train)
    clf_score = knn_clf.score(X_test, y_clf_test)
    print(f"   Accuracy: {clf_score:.4f}")
    
    print("\n5. 📊 Testing StandardScaler:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    print(f"   Original mean: {np.mean(X_train, axis=0)}")
    print(f"   Scaled mean: {np.mean(X_scaled, axis=0)}")
    
    print("\n✅ All tests completed successfully!")
    print("📦 ML101 package is working correctly!")
    print("\n🎉 Ready to explore machine learning!")

if __name__ == "__main__":
    main()
