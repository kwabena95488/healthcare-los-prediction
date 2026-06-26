"""
Visualization utilities for the healthcare analytics project.

This module provides functions for generating various plots and visualizations
for exploratory data analysis (EDA) and model performance analysis.

@module visualize
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_distribution(
    df: pd.DataFrame, 
    column: str, 
    plot_type: str = 'hist', 
    title: Optional[str] = None, 
    filepath: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> None:
    """
    Plots the distribution of a numerical or categorical column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Name of the column to plot.
        plot_type (str): Type of plot ('hist', 'kde', 'box', 'count'). Defaults to 'hist'.
        title (Optional[str]): Title of the plot. Defaults to None.
        filepath (Optional[Union[str, Path]]): Path to save the plot. If None, displays the plot.
        **kwargs: Additional keyword arguments passed to seaborn plotting functions.
        
    @example
    # Plot histogram
    plot_distribution(df, 'Age', plot_type='hist', title='Age Distribution', filepath='results/figures/age_hist.png')
    # Plot countplot for categorical
    plot_distribution(df, 'Gender', plot_type='count', title='Gender Distribution', filepath='results/figures/gender_count.png')
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame.")
        return

    plt.figure(figsize=(10, 6))
    
    if plot_type == 'hist':
        sns.histplot(df[column], kde=True, **kwargs)
    elif plot_type == 'kde':
        sns.kdeplot(df[column], fill=True, **kwargs)
    elif plot_type == 'box':
        sns.boxplot(x=df[column], **kwargs)
    elif plot_type == 'count':
        sns.countplot(x=df[column], **kwargs)
    else:
        logger.warning(f"Unsupported plot_type: {plot_type}. Using 'hist'.")
        sns.histplot(df[column], kde=True, **kwargs)

    plt.title(title if title else f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count' if plot_type in ['hist', 'count'] else 'Density')
    plt.tight_layout()
    
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        logger.info(f"Distribution plot saved to {filepath}")
    else:
        plt.show()
    plt.close()

def plot_correlation_matrix(
    df: pd.DataFrame, 
    title: str = "Correlation Matrix", 
    filepath: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> None:
    """
    Plots a correlation matrix for numerical columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        title (str): Title of the plot. Defaults to "Correlation Matrix".
        filepath (Optional[Union[str, Path]]): Path to save the plot. If None, displays the plot.
        **kwargs: Additional keyword arguments passed to seaborn.heatmap.
        
    @example
    # Plot correlation matrix
    plot_correlation_matrix(df_numerical, title='Feature Correlations', filepath='results/figures/correlation_matrix.png')
    """
    numerical_df = df.select_dtypes(include=np.number)
    if numerical_df.empty:
        logger.warning("No numerical columns found for correlation matrix.")
        return

    corr_matrix = numerical_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, **kwargs)
    plt.title(title)
    plt.tight_layout()
    
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        logger.info(f"Correlation matrix saved to {filepath}")
    else:
        plt.show()
    plt.close()

def plot_roc_curve(
    y_true: Union[pd.Series, np.ndarray], 
    y_prob: Union[pd.Series, np.ndarray], 
    title: str = "ROC Curve", 
    filepath: Optional[Union[str, Path]] = None,
    pos_label: Optional[Any] = None,
    **kwargs: Any
) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve for binary classification.
    
    Args:
        y_true (Union[pd.Series, np.ndarray]): True binary labels.
        y_prob (Union[pd.Series, np.ndarray]): Predicted probabilities of the positive class.
        title (str): Title of the plot. Defaults to "ROC Curve".
        filepath (Optional[Union[str, Path]]): Path to save the plot. If None, displays the plot.
        pos_label (Optional[Any]): The label of the positive class. If None, assumes binary (0, 1) and 1 is positive.
        **kwargs: Additional keyword arguments passed to matplotlib.pyplot.plot.
        
    @example
    # Plot ROC curve
    plot_roc_curve(y_test, y_probabilities[:, 1], title='Model ROC Curve', filepath='results/figures/roc_curve.png')
    """
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        # For multi-class, assume y_prob is for the positive class if pos_label is given
        # Or, if binary, take the probability of the second class
        if pos_label is not None:
            # Need to map pos_label to its encoded integer if y_true is not numeric
            le = LabelEncoder()
            if not np.issubdtype(y_true.dtype, np.number):
                y_true_encoded = le.fit_transform(y_true)
                pos_label_encoded = le.transform([pos_label])[0]
            else:
                y_true_encoded = y_true
                pos_label_encoded = pos_label
            
            # Find the column index for the positive class
            if pos_label_encoded < y_prob.shape[1]:
                y_prob_single_class = y_prob[:, pos_label_encoded]
            else:
                logger.error(f"Positive label {pos_label} (encoded {pos_label_encoded}) out of bounds for y_prob shape {y_prob.shape}.")
                return
        else:
            # Default to probability of class 1 for binary classification
            y_prob_single_class = y_prob[:, 1]
    else:
        y_prob_single_class = y_prob

    fpr, tpr, thresholds = roc_curve(y_true, y_prob_single_class, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})', **kwargs)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        logger.info(f"ROC curve saved to {filepath}")
    else:
        plt.show()
    plt.close()

def plot_precision_recall_curve(
    y_true: Union[pd.Series, np.ndarray], 
    y_prob: Union[pd.Series, np.ndarray], 
    title: str = "Precision-Recall Curve", 
    filepath: Optional[Union[str, Path]] = None,
    pos_label: Optional[Any] = None,
    **kwargs: Any
) -> None:
    """
    Plots the Precision-Recall curve for binary classification.
    
    Args:
        y_true (Union[pd.Series, np.ndarray]): True binary labels.
        y_prob (Union[pd.Series, np.ndarray]): Predicted probabilities of the positive class.
        title (str): Title of the plot. Defaults to "Precision-Recall Curve".
        filepath (Optional[Union[str, Path]]): Path to save the plot. If None, displays the plot.
        pos_label (Optional[Any]): The label of the positive class. If None, assumes binary (0, 1) and 1 is positive.
        **kwargs: Additional keyword arguments passed to matplotlib.pyplot.plot.
        
    @example
    # Plot Precision-Recall curve
    plot_precision_recall_curve(y_test, y_probabilities[:, 1], title='Model PR Curve', filepath='results/figures/pr_curve.png')
    """
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        if pos_label is not None:
            le = LabelEncoder()
            if not np.issubdtype(y_true.dtype, np.number):
                y_true_encoded = le.fit_transform(y_true)
                pos_label_encoded = le.transform([pos_label])[0]
            else:
                y_true_encoded = y_true
                pos_label_encoded = pos_label
            
            if pos_label_encoded < y_prob.shape[1]:
                y_prob_single_class = y_prob[:, pos_label_encoded]
            else:
                logger.error(f"Positive label {pos_label} (encoded {pos_label_encoded}) out of bounds for y_prob shape {y_prob.shape}.")
                return
        else:
            y_prob_single_class = y_prob[:, 1]
    else:
        y_prob_single_class = y_prob

    precision, recall, _ = precision_recall_curve(y_true, y_prob_single_class, pos_label=pos_label)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve', **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        logger.info(f"Precision-Recall curve saved to {filepath}")
    else:
        plt.show()
    plt.close()

# Example usage (for testing this module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulate data
    data = {
        'numerical_feat': np.random.randn(100),
        'categorical_feat': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    }
    df_test = pd.DataFrame(data)
    
    # Simulate binary classification results for ROC/PR curves
    y_true_binary = df_test['target']
    y_prob_binary = np.random.rand(100) # Probabilities for positive class
    
    print("\n--- Testing plot_distribution ---")
    plot_distribution(df_test, 'numerical_feat', plot_type='hist', filepath='results/figures/test_hist.png')
    plot_distribution(df_test, 'categorical_feat', plot_type='count', filepath='results/figures/test_count.png')

    print("\n--- Testing plot_correlation_matrix ---")
    plot_correlation_matrix(df_test, filepath='results/figures/test_corr_matrix.png')

    print("\n--- Testing plot_roc_curve ---")
    plot_roc_curve(y_true_binary, y_prob_binary, filepath='results/figures/test_roc_curve.png')

    print("\n--- Testing plot_precision_recall_curve ---")
    plot_precision_recall_curve(y_true_binary, y_prob_binary, filepath='results/figures/test_pr_curve.png')

    # Simulate multi-class data for ROC/PR curves
    y_true_multi = pd.Series(np.random.randint(0, 3, 100)) # 3 classes
    y_prob_multi = np.random.rand(100, 3) # Probabilities for 3 classes
    y_prob_multi = y_prob_multi / y_prob_multi.sum(axis=1, keepdims=True) # Normalize to sum to 1

    print("\n--- Testing plot_roc_curve (multiclass) ---")
    # For multi-class, you typically plot one-vs-rest or one-vs-one.
    # The function currently handles this by taking pos_label.
    # Let's plot for class 1 as positive.
    plot_roc_curve(y_true_multi, y_prob_multi, pos_label=1, title='Multi-class ROC (Class 1 vs Rest)', filepath='results/figures/test_roc_multi.png')

    print("\n--- Testing plot_precision_recall_curve (multiclass) ---")
    plot_precision_recall_curve(y_true_multi, y_prob_multi, pos_label=1, title='Multi-class PR (Class 1 vs Rest)', filepath='results/figures/test_pr_multi.png')
