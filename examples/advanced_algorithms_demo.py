"""
Advanced ML Algorithms Demonstration

This script demonstrates the usage of Support Vector Machines (SVM),
Principal Component Analysis (PCA), and Random Forest algorithms
implemented in the ML101 project.

Author: ML101 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add algorithm paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms', 'svm'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms', 'pca'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms', 'random_forest'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from svm import SVM, generate_classification_data
from pca import PCA, generate_sample_data as generate_pca_data
from random_forest import RandomForest, generate_sample_data as generate_rf_data
from preprocessing import StandardScaler


def demonstrate_svm():
    """Demonstrate Support Vector Machine functionality."""
    print("=" * 60)
    print("SUPPORT VECTOR MACHINE (SVM) DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_classification_data(n_samples=300, n_features=2, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Test different kernels
    kernels = ['linear', 'rbf', 'poly']
    results = {}
    
    print("\n2. Testing different kernels...")
    for kernel in kernels:
        print(f"\nTesting {kernel.upper()} kernel:")
        
        # Create and train SVM
        svm = SVM(kernel=kernel, C=1.0, random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = svm.score(X_train_scaled, y_train)
        test_score = svm.score(X_test_scaled, y_test)
        
        results[kernel] = {
            'train_score': train_score,
            'test_score': test_score,
            'n_support': svm.n_support_
        }
        
        print(f"  Training accuracy: {train_score:.4f}")
        print(f"  Test accuracy: {test_score:.4f}")
        print(f"  Support vectors: {svm.n_support_}")
        
        # Plot decision boundary for RBF kernel
        if kernel == 'rbf':
            print("  Plotting decision boundary...")
            svm.plot_decision_boundary(X_train_scaled, y_train, 
                                     f"SVM Decision Boundary ({kernel.upper()} kernel)")
    
    # Summary
    print("\n3. Summary:")
    best_kernel = max(results, key=lambda k: results[k]['test_score'])
    print(f"Best kernel: {best_kernel.upper()}")
    print(f"Best test accuracy: {results[best_kernel]['test_score']:.4f}")
    
    return results


def demonstrate_pca():
    """Demonstrate Principal Component Analysis functionality."""
    print("\n" + "=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA) DEMONSTRATION")
    print("=" * 60)
    
    # Generate high-dimensional data
    print("\n1. Generating high-dimensional sample data...")
    X, y = generate_pca_data(n_samples=500, n_features=50, 
                            n_informative=10, random_state=42)
    
    print(f"Original data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of components
    print("\n2. Testing different numbers of components...")
    
    # Fixed number of components
    pca_fixed = PCA(n_components=10)
    X_pca_fixed = pca_fixed.fit_transform(X_scaled)
    
    print(f"PCA with 10 components:")
    print(f"  Reduced shape: {X_pca_fixed.shape}")
    print(f"  Explained variance: {np.sum(pca_fixed.explained_variance_ratio_):.4f}")
    
    # Variance threshold
    pca_var = PCA(n_components=0.95)  # 95% variance
    X_pca_var = pca_var.fit_transform(X_scaled)
    
    print(f"\nPCA with 95% variance:")
    print(f"  Components needed: {X_pca_var.shape[1]}")
    print(f"  Explained variance: {np.sum(pca_var.explained_variance_ratio_):.4f}")
    
    # Reconstruction error
    X_reconstructed = pca_fixed.inverse_transform(X_pca_fixed)
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
    print(f"\nReconstruction error: {reconstruction_error:.6f}")
    
    # Visualizations
    print("\n3. Generating visualizations...")
    
    # Explained variance plot
    pca_vis = PCA(n_components=20)
    pca_vis.fit(X_scaled)
    pca_vis.plot_explained_variance()
    
    # Component visualization
    pca_vis.plot_components(n_components=3)
    
    # 2D projection
    pca_2d = PCA(n_components=2)
    pca_2d.fit(X_scaled)
    pca_2d.plot_2d_projection(X_scaled, y, "PCA 2D Projection")
    
    return {
        'original_shape': X.shape,
        'pca_10_shape': X_pca_fixed.shape,
        'pca_95_components': X_pca_var.shape[1],
        'reconstruction_error': reconstruction_error
    }


def demonstrate_random_forest():
    """Demonstrate Random Forest functionality."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST DEMONSTRATION")
    print("=" * 60)
    
    # Classification example
    print("\n1. CLASSIFICATION EXAMPLE")
    print("-" * 30)
    
    # Generate classification data
    X_class, y_class = generate_rf_data(n_samples=1000, n_features=20, 
                                       n_informative=8, task='classification',
                                       random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X_class))
    X_train, X_test = X_class[:split_idx], X_class[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train Random Forest
    rf_classifier = RandomForest(
        n_estimators=100,
        max_depth=15,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,  # Disable OOB score for now
        random_state=42,
        task='classification'
    )
    
    print("\nTraining Random Forest classifier...")
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate
    train_score = rf_classifier.score(X_train, y_train)
    test_score = rf_classifier.score(X_test, y_test)
    oob_score = rf_classifier.oob_score_
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"OOB score: {oob_score:.4f}")
    
    # Feature importance
    print(f"Top 5 feature importances: {rf_classifier.feature_importances_[:5]}")
    
    # Regression example
    print("\n2. REGRESSION EXAMPLE")
    print("-" * 30)
    
    # Generate regression data
    X_reg, y_reg = generate_rf_data(n_samples=1000, n_features=15, 
                                   n_informative=5, task='regression',
                                   random_state=42)
    
    # Split data
    X_train_reg, X_test_reg = X_reg[:split_idx], X_reg[split_idx:]
    y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
    
    # Train Random Forest
    rf_regressor = RandomForest(
        n_estimators=100,
        max_depth=15,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        task='regression'
    )
    
    print("Training Random Forest regressor...")
    rf_regressor.fit(X_train_reg, y_train_reg)
    
    # Evaluate
    train_r2 = rf_regressor.score(X_train_reg, y_train_reg)
    test_r2 = rf_regressor.score(X_test_reg, y_test_reg)
    oob_r2 = rf_regressor.oob_score_
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"OOB R²: {oob_r2:.4f}")
    
    # Visualizations
    print("\n3. Generating visualizations...")
    
    # Feature importances
    rf_classifier.plot_feature_importances()
    
    # OOB error (simplified)
    rf_classifier.plot_oob_error()
    
    return {
        'classification': {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'oob_score': oob_score
        },
        'regression': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'oob_r2': oob_r2
        }
    }


def demonstrate_combined_workflow():
    """Demonstrate combining PCA with classification algorithms."""
    print("\n" + "=" * 60)
    print("COMBINED WORKFLOW: PCA + CLASSIFICATION")
    print("=" * 60)
    
    # Generate high-dimensional data
    print("\n1. Generating high-dimensional data...")
    X, y = generate_pca_data(n_samples=800, n_features=50, 
                            n_informative=15, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Original data shape: {X.shape}")
    
    # Apply PCA
    print("\n2. Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA reduced shape: {X_train_pca.shape}")
    print(f"Variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Compare algorithms on original vs PCA data
    print("\n3. Comparing algorithms on original vs PCA-reduced data...")
    
    algorithms = {
        'SVM': SVM(kernel='rbf', C=1.0, random_state=42),
        'Random Forest': RandomForest(n_estimators=50, random_state=42, task='classification')
    }
    
    results = {}
    
    for name, algo in algorithms.items():
        print(f"\n{name}:")
        
        # Train on original data
        algo_original = type(algo)(**algo.__dict__)
        algo_original.fit(X_train_scaled, y_train)
        score_original = algo_original.score(X_test_scaled, y_test)
        
        # Train on PCA data
        algo_pca = type(algo)(**algo.__dict__)
        algo_pca.fit(X_train_pca, y_train)
        score_pca = algo_pca.score(X_test_pca, y_test)
        
        results[name] = {
            'original': score_original,
            'pca': score_pca
        }
        
        print(f"  Original data accuracy: {score_original:.4f}")
        print(f"  PCA data accuracy: {score_pca:.4f}")
        print(f"  Difference: {score_pca - score_original:.4f}")
    
    # Summary
    print("\n4. Summary:")
    print("Algorithm performance comparison (Original vs PCA):")
    for name, scores in results.items():
        print(f"  {name}: {scores['original']:.4f} → {scores['pca']:.4f}")
    
    return results


def main():
    """Main demonstration function."""
    print("ML101 - Advanced Algorithms Demonstration")
    print("=" * 60)
    print("This script demonstrates SVM, PCA, and Random Forest algorithms")
    print("implemented from scratch in the ML101 project.")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    svm_results = demonstrate_svm()
    pca_results = demonstrate_pca()
    rf_results = demonstrate_random_forest()
    combined_results = demonstrate_combined_workflow()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("\nSVM Results:")
    for kernel, metrics in svm_results.items():
        print(f"  {kernel.upper()}: {metrics['test_score']:.4f} accuracy")
    
    print(f"\nPCA Results:")
    print(f"  Original dimensions: {pca_results['original_shape'][1]}")
    print(f"  PCA (10 components): {pca_results['pca_10_shape'][1]}")
    print(f"  PCA (95% variance): {pca_results['pca_95_components']}")
    
    print(f"\nRandom Forest Results:")
    print(f"  Classification accuracy: {rf_results['classification']['test_accuracy']:.4f}")
    print(f"  Regression R²: {rf_results['regression']['test_r2']:.4f}")
    
    print(f"\nCombined Workflow:")
    for algo, scores in combined_results.items():
        print(f"  {algo}: {scores['original']:.4f} → {scores['pca']:.4f}")
    
    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
