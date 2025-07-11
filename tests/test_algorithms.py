"""
Test suite for ML101 implementations

This module contains unit tests for all machine learning algorithm implementations.
"""

import pytest
import numpy as np
from ml101.linear_models.linear_regression import LinearRegression, generate_linear_data
from ml101.linear_models.logistic_regression import LogisticRegression, generate_classification_data
from ml101.neighbors.knn import KNearestNeighbors, generate_knn_data
from ml101.svm.svm import SVM, generate_classification_data as svm_generate_data
from ml101.decomposition.pca import PCA, generate_sample_data as pca_generate_data
from ml101.ensemble.random_forest import RandomForest, generate_sample_data as rf_generate_data
from ml101.linear_models.ridge_regression import RidgeRegression, generate_regression_data
from ml101.linear_models.lasso_regression import LassoRegression, generate_sparse_regression_data
from ml101.utils.metrics import ClassificationMetrics, RegressionMetrics
from ml101.utils.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, train_test_split


class TestLinearRegression:
    """Test cases for Linear Regression implementation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X, self.y = generate_linear_data(n_samples=100, noise=0.1, random_state=42)
    
    def test_normal_equation_method(self):
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
    
    def test_gradient_descent_method(self):
        """Test linear regression with gradient descent."""
        model = LinearRegression(method='gradient_descent', learning_rate=0.01, max_iterations=1000)
        model.fit(self.X, self.y)
        
        # Check that model was fitted
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.cost_history) > 0
        
        # Check predictions
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        
        # Check R² score is reasonable
        r2 = model.score(self.X, self.y)
        assert r2 > 0.5
    
    def test_methods_give_similar_results(self):
        """Test that both methods give similar results."""
        model_normal = LinearRegression(method='normal')
        model_gd = LinearRegression(method='gradient_descent', learning_rate=0.01, max_iterations=1000)
        
        model_normal.fit(self.X, self.y)
        model_gd.fit(self.X, self.y)
        
        # Weights should be similar
        assert np.allclose(model_normal.weights, model_gd.weights, atol=0.1)
        assert np.allclose(model_normal.bias, model_gd.bias, atol=0.1)
    
    def test_multivariable_regression(self):
        """Test linear regression with multiple features."""
        np.random.seed(42)
        X_multi = np.random.randn(100, 3)
        y_multi = X_multi @ np.array([1.5, -2.0, 0.8]) + 0.5 + np.random.normal(0, 0.1, 100)
        
        model = LinearRegression(method='normal')
        model.fit(X_multi, y_multi)
        
        assert len(model.weights) == 3
        r2 = model.score(X_multi, y_multi)
        assert r2 > 0.8


class TestLogisticRegression:
    """Test cases for Logistic Regression implementation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X, self.y = generate_classification_data(n_samples=200, n_classes=2, random_state=42)
    
    def test_binary_classification(self):
        """Test binary logistic regression."""
        model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
        model.fit(self.X, self.y)
        
        # Check that model was fitted
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.cost_history) > 0
        
        # Check predictions
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        assert set(predictions) <= set(self.y)  # Predictions should be subset of original classes
        
        # Check probability predictions
        probabilities = model.predict_proba(self.X)
        assert probabilities.shape == (len(self.y), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities should sum to 1
        
        # Check accuracy is reasonable
        accuracy = model.score(self.X, self.y)
        assert accuracy > 0.6
    
    def test_multiclass_classification(self):
        """Test multiclass logistic regression."""
        X_multi, y_multi = generate_classification_data(n_samples=300, n_classes=3, random_state=42)
        
        model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
        model.fit(X_multi, y_multi)
        
        # Check predictions
        predictions = model.predict(X_multi)
        assert len(predictions) == len(y_multi)
        assert set(predictions) <= set(y_multi)
        
        # Check probability predictions
        probabilities = model.predict_proba(X_multi)
        assert probabilities.shape == (len(y_multi), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Check accuracy is reasonable
        accuracy = model.score(X_multi, y_multi)
        assert accuracy > 0.5
    
    def test_convergence(self):
        """Test that cost decreases during training."""
        model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
        model.fit(self.X, self.y)
        
        # Cost should generally decrease
        cost_history = model.cost_history
        assert len(cost_history) > 10
        assert cost_history[-1] < cost_history[0]  # Final cost should be less than initial


class TestKNearestNeighbors:
    """Test cases for K-Nearest Neighbors implementation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_clf, self.y_clf = generate_knn_data(task='classification', n_samples=150, random_state=42)
        self.X_reg, self.y_reg = generate_knn_data(task='regression', n_samples=100, random_state=42)
    
    def test_classification(self):
        """Test KNN classification."""
        model = KNearestNeighbors(k=5, task='classification')
        model.fit(self.X_clf, self.y_clf)
        
        # Check predictions
        predictions = model.predict(self.X_clf)
        assert len(predictions) == len(self.y_clf)
        assert set(predictions) <= set(self.y_clf)
        
        # Check probability predictions
        probabilities = model.predict_proba(self.X_clf)
        assert probabilities.shape[0] == len(self.y_clf)
        assert probabilities.shape[1] == len(np.unique(self.y_clf))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Check accuracy is reasonable
        accuracy = model.score(self.X_clf, self.y_clf)
        assert accuracy > 0.5
    
    def test_regression(self):
        """Test KNN regression."""
        model = KNearestNeighbors(k=5, task='regression')
        model.fit(self.X_reg, self.y_reg)
        
        # Check predictions
        predictions = model.predict(self.X_reg)
        assert len(predictions) == len(self.y_reg)
        assert predictions.dtype == np.float64
        
        # Check R² score is reasonable
        r2 = model.score(self.X_reg, self.y_reg)
        assert r2 > 0.3
    
    def test_different_k_values(self):
        """Test KNN with different k values."""
        k_values = [1, 3, 5, 10]
        accuracies = []
        
        for k in k_values:
            model = KNearestNeighbors(k=k, task='classification')
            model.fit(self.X_clf, self.y_clf)
            accuracy = model.score(self.X_clf, self.y_clf)
            accuracies.append(accuracy)
        
        # k=1 should have perfect training accuracy
        assert accuracies[0] == 1.0
        
        # All accuracies should be reasonable
        assert all(acc > 0.5 for acc in accuracies)
    
    def test_distance_metrics(self):
        """Test different distance metrics."""
        metrics = ['euclidean', 'manhattan', 'minkowski']
        
        for metric in metrics:
            model = KNearestNeighbors(k=5, task='classification', distance_metric=metric)
            model.fit(self.X_clf, self.y_clf)
            accuracy = model.score(self.X_clf, self.y_clf)
            assert accuracy > 0.5


class TestMetrics:
    """Test cases for metrics implementations."""
    
    def setup_method(self):
        """Set up test data."""
        self.y_true_clf = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 1])
        self.y_pred_clf = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 2])
        
        self.y_true_reg = np.array([3.0, -0.5, 2.0, 7.0, 1.0])
        self.y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0, 1.2])
    
    def test_classification_metrics(self):
        """Test classification metrics."""
        # Accuracy
        accuracy = ClassificationMetrics.accuracy_score(self.y_true_clf, self.y_pred_clf)
        expected_accuracy = 7 / 10  # 6 correct predictions out of 10
        assert accuracy == expected_accuracy
        
        # Confusion matrix
        cm = ClassificationMetrics.confusion_matrix(self.y_true_clf, self.y_pred_clf)
        assert cm.shape == (3, 3)  # 3 classes
        assert cm.sum() == len(self.y_true_clf)
        
        # Precision, recall, F1-score
        precision, recall, fscore = ClassificationMetrics.precision_recall_fscore(
            self.y_true_clf, self.y_pred_clf, average='macro')
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= fscore <= 1
    
    def test_regression_metrics(self):
        """Test regression metrics."""
        # MSE
        mse = RegressionMetrics.mean_squared_error(self.y_true_reg, self.y_pred_reg)
        expected_mse = np.mean((self.y_true_reg - self.y_pred_reg) ** 2)
        assert np.isclose(mse, expected_mse)
        
        # RMSE
        rmse = RegressionMetrics.root_mean_squared_error(self.y_true_reg, self.y_pred_reg)
        assert np.isclose(rmse, np.sqrt(expected_mse))
        
        # MAE
        mae = RegressionMetrics.mean_absolute_error(self.y_true_reg, self.y_pred_reg)
        expected_mae = np.mean(np.abs(self.y_true_reg - self.y_pred_reg))
        assert np.isclose(mae, expected_mae)
        
        # R²
        r2 = RegressionMetrics.r2_score(self.y_true_reg, self.y_pred_reg)
        assert -1 <= r2 <= 1  # R² can be negative for very bad models


class TestPreprocessing:
    """Test cases for preprocessing utilities."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 3) * [10, 5, 2] + [50, 20, 5]
        self.y = np.random.choice(['A', 'B', 'C'], 100)
    
    def test_standard_scaler(self):
        """Test StandardScaler."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Mean should be close to 0
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        
        # Std should be close to 1
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)
        
        # Test inverse transform
        X_inverse = scaler.inverse_transform(X_scaled)
        assert np.allclose(X_inverse, self.X)
    
    def test_minmax_scaler(self):
        """Test MinMaxScaler."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Min should be 0, max should be 1
        assert np.allclose(np.min(X_scaled, axis=0), 0)
        assert np.allclose(np.max(X_scaled, axis=0), 1)
        
        # Test inverse transform
        X_inverse = scaler.inverse_transform(X_scaled)
        assert np.allclose(X_inverse, self.X)
    
    def test_label_encoder(self):
        """Test LabelEncoder."""
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(self.y)
        
        # Encoded labels should be 0, 1, 2
        assert set(y_encoded) == {0, 1, 2}
        
        # Test inverse transform
        y_inverse = encoder.inverse_transform(y_encoded)
        assert np.array_equal(y_inverse, self.y)
    
    def test_train_test_split(self):
        """Test train_test_split function."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
        
        # Check sizes
        assert len(X_train) == 70
        assert len(X_test) == 30
        assert len(y_train) == 70
        assert len(y_test) == 30
        
        # Check that no data is lost
        total_samples = len(X_train) + len(X_test)
        assert total_samples == len(self.X)


class TestSVM:
    """Test cases for Support Vector Machine implementation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X, self.y = svm_generate_data(n_samples=100, n_features=2, random_state=42)
    
    def test_linear_kernel(self):
        """Test SVM with linear kernel."""
        model = SVM(kernel='linear', C=1.0, random_state=42)
        model.fit(self.X, self.y)
        
        # Check that model is trained
        assert model.support_vectors_ is not None
        assert model.dual_coef_ is not None
        assert model.intercept_ is not None
        
        # Test prediction
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        assert set(predictions) == set(model.classes_)
    
    def test_rbf_kernel(self):
        """Test SVM with RBF kernel."""
        model = SVM(kernel='rbf', C=1.0, random_state=42)
        model.fit(self.X, self.y)
        
        # Test prediction
        predictions = model.predict(self.X)
        accuracy = model.score(self.X, self.y)
        assert accuracy > 0.5  # Should be better than random
    
    def test_predict_proba(self):
        """Test probability predictions."""
        model = SVM(kernel='rbf', C=1.0, random_state=42)
        model.fit(self.X, self.y)
        
        probabilities = model.predict_proba(self.X)
        assert probabilities.shape == (len(self.X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestPCA:
    """Test cases for Principal Component Analysis implementation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X, self.y = pca_generate_data(n_samples=100, n_features=10, random_state=42)
    
    def test_fit_transform(self):
        """Test PCA fit and transform."""
        model = PCA(n_components=5)
        X_transformed = model.fit_transform(self.X)
        
        # Check shapes
        assert X_transformed.shape == (100, 5)
        assert model.components_.shape == (5, 10)
        assert len(model.explained_variance_ratio_) == 5
    
    def test_variance_threshold(self):
        """Test PCA with variance threshold."""
        model = PCA(n_components=0.9)  # 90% variance
        X_transformed = model.fit_transform(self.X)
        
        # Check that variance is preserved
        total_variance = np.sum(model.explained_variance_ratio_)
        assert total_variance >= 0.9
    
    def test_inverse_transform(self):
        """Test PCA inverse transform."""
        model = PCA(n_components=5)
        X_transformed = model.fit_transform(self.X)
        X_reconstructed = model.inverse_transform(X_transformed)
        
        # Check shape
        assert X_reconstructed.shape == self.X.shape
        
        # Check that some information is preserved
        reconstruction_error = np.mean((self.X - X_reconstructed) ** 2)
        assert reconstruction_error < 1.0  # Should be reasonable


class TestRandomForest:
    """Test cases for Random Forest implementation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_class, self.y_class = rf_generate_data(n_samples=200, n_features=10, 
                                                     task='classification', random_state=42)
        self.X_reg, self.y_reg = rf_generate_data(n_samples=200, n_features=10, 
                                                 task='regression', random_state=42)
    
    def test_classification(self):
        """Test Random Forest classification."""
        model = RandomForest(n_estimators=10, random_state=42, task='classification')
        model.fit(self.X_class, self.y_class)
        
        # Check that model is trained
        assert len(model.estimators_) == 10
        assert model.feature_importances_ is not None
        
        # Test prediction
        predictions = model.predict(self.X_class)
        assert len(predictions) == len(self.y_class)
        
        # Test accuracy
        accuracy = model.score(self.X_class, self.y_class)
        assert accuracy > 0.5  # Should be better than random
    
    def test_regression(self):
        """Test Random Forest regression."""
        model = RandomForest(n_estimators=10, random_state=42, task='regression')
        model.fit(self.X_reg, self.y_reg)
        
        # Test prediction
        predictions = model.predict(self.X_reg)
        assert len(predictions) == len(self.y_reg)
        
        # Test R² score
        r2_score = model.score(self.X_reg, self.y_reg)
        assert r2_score > 0.0  # Should explain some variance
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        model = RandomForest(n_estimators=10, random_state=42, task='classification')
        model.fit(self.X_class, self.y_class)
        
        # Check feature importances
        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == self.X_class.shape[1]
        assert np.allclose(np.sum(model.feature_importances_), 1.0)  # Should sum to 1
    
    def test_oob_score(self):
        """Test out-of-bag score calculation."""
        model = RandomForest(n_estimators=10, oob_score=True, random_state=42, task='classification')
        model.fit(self.X_class, self.y_class)
        
        # Check that OOB score is calculated
        assert model.oob_score_ is not None
        assert 0 <= model.oob_score_ <= 1


class TestRidgeRegression:
    """Test cases for Ridge Regression implementation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X, self.y = generate_regression_data(n_samples=100, n_features=5, noise=0.1, random_state=42)
    
    def test_normal_equation_solver(self):
        """Test Ridge regression with normal equation solver."""
        model = RidgeRegression(alpha=1.0, solver='normal')
        model.fit(self.X, self.y)
        
        # Check that model was fitted
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert len(model.coef_) == self.X.shape[1]
        
        # Check predictions
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        
        # Check R² score is reasonable
        r2 = model.score(self.X, self.y)
        assert r2 > 0.5  # Should have decent fit on synthetic data
    
    def test_gradient_descent_solver(self):
        """Test Ridge regression with gradient descent solver."""
        model = RidgeRegression(alpha=1.0, solver='gradient_descent', max_iter=1000)
        model.fit(self.X, self.y)
        
        # Check that model was fitted
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert len(model.cost_history_) > 0
        
        # Check predictions
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        
        # Check R² score is reasonable
        r2 = model.score(self.X, self.y)
        assert r2 > 0.5
    
    def test_regularization_effect(self):
        """Test that regularization reduces coefficient magnitude."""
        # Compare with no regularization (alpha=0)
        model_no_reg = RidgeRegression(alpha=0.0, solver='normal')
        model_reg = RidgeRegression(alpha=10.0, solver='normal')
        
        model_no_reg.fit(self.X, self.y)
        model_reg.fit(self.X, self.y)
        
        # Regularized model should have smaller coefficient norm
        norm_no_reg = np.linalg.norm(model_no_reg.coef_)
        norm_reg = np.linalg.norm(model_reg.coef_)
        
        assert norm_reg < norm_no_reg


class TestLassoRegression:
    """Test cases for Lasso Regression implementation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X, self.y, self.true_coef = generate_sparse_regression_data(
            n_samples=100, n_features=10, n_informative=3, noise=0.1, random_state=42
        )
    
    def test_coordinate_descent_fitting(self):
        """Test Lasso regression fitting with coordinate descent."""
        model = LassoRegression(alpha=0.1, max_iter=1000)
        model.fit(self.X, self.y)
        
        # Check that model was fitted
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert len(model.coef_) == self.X.shape[1]
        assert model.n_iter_ is not None
        
        # Check predictions
        predictions = model.predict(self.X)
        assert len(predictions) == len(self.y)
        
        # Check R² score is reasonable
        r2 = model.score(self.X, self.y)
        assert r2 > 0.3  # Lasso might have lower R² due to sparsity
    
    def test_feature_selection(self):
        """Test that Lasso performs feature selection."""
        model = LassoRegression(alpha=0.1, max_iter=1000)
        model.fit(self.X, self.y)
        
        # Check that some coefficients are zero (feature selection)
        n_zero_coef = np.sum(np.abs(model.coef_) < 1e-6)
        assert n_zero_coef > 0  # Should eliminate some features
        
        # Check selected features
        selected_features = model.get_selected_features()
        assert len(selected_features) < self.X.shape[1]  # Should select fewer than total
        assert len(selected_features) > 0  # Should select at least one feature
    
    def test_sparsity_increases_with_alpha(self):
        """Test that higher alpha values lead to more sparse solutions."""
        model_low_alpha = LassoRegression(alpha=0.01, max_iter=1000)
        model_high_alpha = LassoRegression(alpha=1.0, max_iter=1000)
        
        model_low_alpha.fit(self.X, self.y)
        model_high_alpha.fit(self.X, self.y)
        
        # Higher alpha should lead to more sparse solution
        n_selected_low = len(model_low_alpha.get_selected_features())
        n_selected_high = len(model_high_alpha.get_selected_features())
        
        assert n_selected_high <= n_selected_low
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        model = LassoRegression(alpha=0.1, max_iter=1000)
        model.fit(self.X, self.y)
        
        importance = model.get_feature_importance()
        assert len(importance) == self.X.shape[1]
        assert np.all(importance >= 0)  # Importance should be non-negative
        assert np.isclose(np.sum(importance), 1.0) or np.sum(importance) == 0.0  # Should sum to 1 or all zero

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
