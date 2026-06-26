"""
Evaluation metrics for the healthcare analytics project.

This module provides functions for calculating various performance metrics
for classification models.

@module metrics
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)

def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray], 
    y_pred: Union[pd.Series, np.ndarray], 
    y_prob: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    average_method: str = 'weighted',
    labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculates a comprehensive set of classification metrics.
    
    Args:
        y_true (Union[pd.Series, np.ndarray]): True labels.
        y_pred (Union[pd.Series, np.ndarray]): Predicted labels.
        y_prob (Optional[Union[pd.DataFrame, np.ndarray]]): Predicted probabilities for each class.
                                                            Required for ROC AUC and log_loss.
                                                            Shape should be (n_samples, n_classes).
        average_method (str): Averaging method for precision, recall, F1-score ('weighted', 'macro', 'micro').
                              Defaults to 'weighted'.
        labels (Optional[List[Any]]): The set of labels to include when average is 'binary',
                                      and their order if average is None.
        target_names (Optional[List[str]]): Names of the target classes for logging.
                                            
    Returns:
        Dict[str, float]: A dictionary of calculated metrics.
        
    @example
    metrics = calculate_classification_metrics(y_test, y_predictions, y_probabilities)
    print(metrics)
    """
    metrics = {}
    
    # Ensure labels are numeric for scikit-learn metrics if they are not already
    le = LabelEncoder()
    if not np.issubdtype(y_true.dtype, np.number):
        y_true_encoded = le.fit_transform(y_true)
        y_pred_encoded = le.transform(y_pred) # Use transform for y_pred
        if labels is None:
            labels_encoded = le.transform(le.classes_)
        else:
            labels_encoded = le.transform(labels)
    else:
        y_true_encoded = y_true
        y_pred_encoded = y_pred
        labels_encoded = labels if labels is not None else np.unique(y_true_encoded)

    metrics['accuracy'] = accuracy_score(y_true_encoded, y_pred_encoded)
    metrics[f'precision_{average_method}'] = precision_score(y_true_encoded, y_pred_encoded, average=average_method, labels=labels_encoded, zero_division=0)
    metrics[f'recall_{average_method}'] = recall_score(y_true_encoded, y_pred_encoded, average=average_method, labels=labels_encoded, zero_division=0)
    metrics[f'f1_score_{average_method}'] = f1_score(y_true_encoded, y_pred_encoded, average=average_method, labels=labels_encoded, zero_division=0)
    
    if y_prob is not None:
        num_classes = len(np.unique(y_true_encoded))
        if num_classes > 2:
            # Multi-class ROC AUC
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true_encoded, y_prob, multi_class='ovr', average=average_method, labels=labels_encoded)
                metrics['roc_auc_ovo'] = roc_auc_score(y_true_encoded, y_prob, multi_class='ovo', average=average_method, labels=labels_encoded)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC AUC score: {e}. Ensure y_prob has correct shape and labels are consistent.")
                metrics['roc_auc_ovr'] = np.nan
                metrics['roc_auc_ovo'] = np.nan
            # Log Loss
            try:
                metrics['log_loss'] = log_loss(y_true_encoded, y_prob, labels=labels_encoded)
            except ValueError as e:
                logger.warning(f"Could not calculate log_loss: {e}. Ensure y_prob has correct shape and labels are consistent.")
                metrics['log_loss'] = np.nan

        elif num_classes == 2:
            # Binary ROC AUC (expects probabilities of the positive class)
            try:
                # Assuming y_prob[:, 1] is the probability of the positive class
                metrics['roc_auc'] = roc_auc_score(y_true_encoded, y_prob[:, 1] if y_prob.ndim > 1 else y_prob, labels=labels_encoded)
                metrics['log_loss'] = log_loss(y_true_encoded, y_prob, labels=labels_encoded)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC AUC/log_loss for binary case: {e}.")
                metrics['roc_auc'] = np.nan
                metrics['log_loss'] = np.nan
        else: # Single class
            metrics['roc_auc'] = np.nan
            metrics['log_loss'] = np.nan


    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true_encoded, y_pred_encoded)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true_encoded, y_pred_encoded)
    
    logger.info("Calculated classification metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  - {metric_name}: {value:.4f}")
        
    return metrics

def get_confusion_matrix(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], labels: Optional[List[Any]] = None) -> np.ndarray:
    """
    Generates the confusion matrix.
    
    Args:
        y_true (Union[pd.Series, np.ndarray]): True labels.
        y_pred (Union[pd.Series, np.ndarray]): Predicted labels.
        labels (Optional[List[Any]]): List of labels to index the matrix.
                                      This may be used to reorder or select a subset of labels.
                                      If None, those that appear at least once in y_true or y_pred are used
                                      in sorted order.
                                      
    Returns:
        np.ndarray: Confusion matrix.
        
    @example
    cm = get_confusion_matrix(y_test, y_predictions)
    print(cm)
    """
    # Ensure labels are numeric for scikit-learn metrics if they are not already
    le = LabelEncoder()
    if not np.issubdtype(y_true.dtype, np.number):
        y_true_encoded = le.fit_transform(y_true)
        y_pred_encoded = le.transform(y_pred)
        labels_encoded = le.transform(labels) if labels is not None else None
    else:
        y_true_encoded = y_true
        y_pred_encoded = y_pred
        labels_encoded = labels

    cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=labels_encoded)
    logger.info("Generated confusion matrix.")
    return cm

# Example usage (for testing this module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulate data for multiclass classification
    y_true_multi = pd.Series(['cat', 'dog', 'cat', 'fish', 'dog', 'fish', 'cat', 'dog', 'fish', 'cat'])
    y_pred_multi = pd.Series(['cat', 'cat', 'cat', 'fish', 'dog', 'cat', 'fish', 'dog', 'fish', 'dog'])
    # Probabilities: rows are samples, columns are classes (cat, dog, fish) in alphabetical order
    y_prob_multi = np.array([
        [0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1],
        [0.4, 0.2, 0.4], [0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1]
    ])
    
    print("\n--- Testing calculate_classification_metrics (multiclass) ---")
    metrics_multi = calculate_classification_metrics(y_true_multi, y_pred_multi, y_prob_multi, target_names=['cat', 'dog', 'fish'])
    print(metrics_multi)

    print("\n--- Testing get_confusion_matrix (multiclass) ---")
    cm_multi = get_confusion_matrix(y_true_multi, y_pred_multi, labels=['cat', 'dog', 'fish'])
    print(cm_multi)

    # Simulate data for binary classification
    y_true_binary = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    y_pred_binary = pd.Series([0, 0, 0, 1, 0, 0, 1, 1, 1, 0])
    # Probabilities: rows are samples, columns are classes (0, 1)
    y_prob_binary = np.array([
        [0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3],
        [0.55, 0.45], [0.4, 0.6], [0.1, 0.9], [0.3, 0.7], [0.95, 0.05]
    ])

    print("\n--- Testing calculate_classification_metrics (binary) ---")
    metrics_binary = calculate_classification_metrics(y_true_binary, y_pred_binary, y_prob_binary, target_names=['class_0', 'class_1'])
    print(metrics_binary)

    print("\n--- Testing get_confusion_matrix (binary) ---")
    cm_binary = get_confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    print(cm_binary)
