"""
Machine Learning Metrics Implementation

This module implements common evaluation metrics for classification and regression
from scratch.
"""

import numpy as np
from typing import Optional, Union, List
import matplotlib.pyplot as plt
from collections import Counter


class ClassificationMetrics:
    """
    Classification metrics implementation from scratch.
    """
    
    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        labels: Optional[List] = None) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        labels : List, optional
            List of labels to index the matrix
            
        Returns:
        --------
        cm : np.ndarray
            Confusion matrix
        """
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        
        n_labels = len(labels)
        label_to_ind = {label: i for i, label in enumerate(labels)}
        
        cm = np.zeros((n_labels, n_labels), dtype=int)
        
        for true_label, pred_label in zip(y_true, y_pred):
            cm[label_to_ind[true_label], label_to_ind[pred_label]] += 1
        
        return cm
    
    @staticmethod
    def precision_recall_fscore(y_true: np.ndarray, y_pred: np.ndarray, 
                               average: str = 'binary', pos_label: Union[str, int] = 1) -> tuple:
        """
        Calculate precision, recall, and F1-score.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        average : str
            Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        pos_label : Union[str, int]
            Positive label for binary classification
            
        Returns:
        --------
        precision, recall, fscore : tuple
            Precision, recall, and F1-score
        """
        if average == 'binary':
            return ClassificationMetrics._binary_precision_recall_fscore(y_true, y_pred, pos_label)
        
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
        if average == 'micro':
            return ClassificationMetrics._micro_precision_recall_fscore(y_true, y_pred, labels)
        elif average in ['macro', 'weighted']:
            return ClassificationMetrics._macro_weighted_precision_recall_fscore(
                y_true, y_pred, labels, average)
        else:
            raise ValueError(f"Unknown average type: {average}")
    
    @staticmethod
    def _binary_precision_recall_fscore(y_true: np.ndarray, y_pred: np.ndarray, 
                                       pos_label: Union[str, int]) -> tuple:
        """Calculate binary classification metrics."""
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, fscore
    
    @staticmethod
    def _micro_precision_recall_fscore(y_true: np.ndarray, y_pred: np.ndarray, 
                                      labels: List) -> tuple:
        """Calculate micro-averaged metrics."""
        tp_sum = fp_sum = fn_sum = 0
        
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, fscore
    
    @staticmethod
    def _macro_weighted_precision_recall_fscore(y_true: np.ndarray, y_pred: np.ndarray, 
                                               labels: List, average: str) -> tuple:
        """Calculate macro or weighted averaged metrics."""
        precisions, recalls, fscores = [], [], []
        weights = []
        
        for label in labels:
            p, r, f = ClassificationMetrics._binary_precision_recall_fscore(y_true, y_pred, label)
            precisions.append(p)
            recalls.append(r)
            fscores.append(f)
            
            if average == 'weighted':
                weights.append(np.sum(y_true == label))
        
        if average == 'macro':
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            fscore = np.mean(fscores)
        else:  # weighted
            total_weight = sum(weights)
            precision = np.average(precisions, weights=weights) if total_weight > 0 else 0.0
            recall = np.average(recalls, weights=weights) if total_weight > 0 else 0.0
            fscore = np.average(fscores, weights=weights) if total_weight > 0 else 0.0
        
        return precision, recall, fscore
    
    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: Optional[List] = None, 
                            target_names: Optional[List[str]] = None) -> str:
        """
        Generate a classification report.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        labels : List, optional
            Labels to include in the report
        target_names : List[str], optional
            Display names for the labels
            
        Returns:
        --------
        report : str
            Text summary of precision, recall, F1-score for each class
        """
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        
        if target_names is None:
            target_names = [str(label) for label in labels]
        
        # Calculate metrics for each class
        report_dict = {}
        for i, (label, name) in enumerate(zip(labels, target_names)):
            p, r, f = ClassificationMetrics._binary_precision_recall_fscore(y_true, y_pred, label)
            support = np.sum(y_true == label)
            report_dict[name] = {'precision': p, 'recall': r, 'f1-score': f, 'support': support}
        
        # Calculate overall metrics
        accuracy = ClassificationMetrics.accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f = ClassificationMetrics.precision_recall_fscore(
            y_true, y_pred, average='macro')
        weighted_p, weighted_r, weighted_f = ClassificationMetrics.precision_recall_fscore(
            y_true, y_pred, average='weighted')
        
        # Format report
        width = max(len(name) for name in target_names + ['weighted avg'])
        head_fmt = f"{{:>{width}s}} {{:>9}} {{:>9}} {{:>9}} {{:>9}}"
        row_fmt = f"{{:>{width}s}} {{:>9.2f}} {{:>9.2f}} {{:>9.2f}} {{:>9}}"
        
        report = head_fmt.format('', 'precision', 'recall', 'f1-score', 'support') + '\n\n'
        
        for name in target_names:
            values = report_dict[name]
            report += row_fmt.format(name, values['precision'], values['recall'], 
                                   values['f1-score'], values['support']) + '\n'
        
        report += '\n'
        report += row_fmt.format('accuracy', '', '', accuracy, len(y_true)) + '\n'
        report += row_fmt.format('macro avg', macro_p, macro_r, macro_f, len(y_true)) + '\n'
        report += row_fmt.format('weighted avg', weighted_p, weighted_r, weighted_f, len(y_true))
        
        return report


class RegressionMetrics:
    """
    Regression metrics implementation from scratch.
    """
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(RegressionMetrics.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² (coefficient of determination)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    @staticmethod
    def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """Calculate Adjusted R²."""
        r2 = RegressionMetrics.r2_score(y_true, y_pred)
        n = len(y_true)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class MetricsVisualization:
    """
    Visualization utilities for metrics.
    """
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: List[str], 
                            title: str = 'Confusion Matrix', 
                            normalize: bool = False) -> None:
        """
        Plot confusion matrix as a heatmap.
        
        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix
        labels : List[str]
            Class labels
        title : str
            Plot title
        normalize : bool
            Whether to normalize the confusion matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = 'Regression Results') -> None:
        """
        Plot regression results with actual vs predicted and residuals.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        title : str
            Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Calculate and display metrics
        r2 = RegressionMetrics.r2_score(y_true, y_pred)
        rmse = RegressionMetrics.root_mean_squared_error(y_true, y_pred)
        mae = RegressionMetrics.mean_absolute_error(y_true, y_pred)
        
        axes[0].text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def model_evaluation_example():
    """Example of using the metrics classes."""
    print("METRICS EVALUATION EXAMPLE")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    
    # Classification example
    print("\n--- CLASSIFICATION METRICS ---")
    y_true_clf = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 1])
    y_pred_clf = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 2])
    
    # Calculate metrics
    accuracy = ClassificationMetrics.accuracy_score(y_true_clf, y_pred_clf)
    cm = ClassificationMetrics.confusion_matrix(y_true_clf, y_pred_clf)
    precision, recall, fscore = ClassificationMetrics.precision_recall_fscore(
        y_true_clf, y_pred_clf, average='macro')
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro Precision: {precision:.3f}")
    print(f"Macro Recall: {recall:.3f}")
    print(f"Macro F1-Score: {fscore:.3f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    report = ClassificationMetrics.classification_report(
        y_true_clf, y_pred_clf, target_names=['Class A', 'Class B', 'Class C'])
    print(report)
    
    # Regression example
    print("\n--- REGRESSION METRICS ---")
    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0, 1.0])
    y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0, 1.2])
    
    mse = RegressionMetrics.mean_squared_error(y_true_reg, y_pred_reg)
    rmse = RegressionMetrics.root_mean_squared_error(y_true_reg, y_pred_reg)
    mae = RegressionMetrics.mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = RegressionMetrics.r2_score(y_true_reg, y_pred_reg)
    
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")


if __name__ == "__main__":
    model_evaluation_example()
