"""
Model interpretability utilities for the healthcare analytics project.

This module provides functions for analyzing and explaining model predictions,
including SHAP values and permutation importance.

@module interpretability
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)

def plot_shap_summary(
    model: Any, 
    X: pd.DataFrame, 
    plot_type: str = "bar", 
    title: str = "SHAP Feature Importance", 
    filepath: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> None:
    """
    Generates and plots SHAP summary plot (bar or beeswarm).
    
    Args:
        model (Any): Trained model (e.g., scikit-learn, XGBoost, LightGBM, CatBoost).
        X (pd.DataFrame): Input features DataFrame used for SHAP explanation.
        plot_type (str): Type of SHAP plot ('bar' or 'beeswarm'). Defaults to 'bar'.
        title (str): Title of the plot. Defaults to "SHAP Feature Importance".
        filepath (Optional[Union[str, Path]]): Path to save the plot. If None, displays the plot.
        **kwargs: Additional keyword arguments passed to shap.summary_plot.
        
    @example
    # Plot SHAP bar summary
    plot_shap_summary(model, X_test_sample, plot_type='bar', filepath="results/figures/shap_bar.png")
    
    # Plot SHAP beeswarm summary
    plot_shap_summary(model, X_test_sample, plot_type='beeswarm', filepath="results/figures/shap_beeswarm.png")
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type=plot_type, show=False, **kwargs)
        plt.title(title)
        plt.tight_layout()
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath)
            logger.info(f"SHAP summary plot saved to {filepath}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate SHAP summary plot: {e}")

def calculate_permutation_importance(
    model: Any, 
    X: pd.DataFrame, 
    y: pd.Series, 
    metric: str = 'accuracy', 
    n_repeats: int = 5, 
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Calculates permutation importance for features.
    
    Args:
        model (Any): Trained model.
        X (pd.DataFrame): Validation/test features.
        y (pd.Series): True labels for validation/test set.
        metric (str): Metric to use for evaluation (e.g., 'accuracy', 'f1_weighted').
        n_repeats (int): Number of times to permute a feature. Defaults to 5.
        random_state (Optional[int]): Seed for reproducibility. Defaults to 42.
        
    Returns:
        pd.DataFrame: DataFrame with feature importances and their standard deviations.
        
    @example
    # Calculate permutation importance
    importance_df = calculate_permutation_importance(model, X_test, y_test, metric='f1_weighted')
    print(importance_df)
    """
    from sklearn.inspection import permutation_importance
    
    try:
        result = permutation_importance(
            model, X, y, 
            scoring=metric, 
            n_repeats=n_repeats, 
            random_state=random_state, 
            n_jobs=-1
        )
        
        sorted_idx = result.importances_mean.argsort()[::-1]
        importance_df = pd.DataFrame({
            'Feature': X.columns[sorted_idx],
            'Importance_Mean': result.importances_mean[sorted_idx],
            'Importance_Std': result.importances_std[sorted_idx]
        })
        logger.info("Calculated permutation importance.")
        return importance_df
    except Exception as e:
        logger.error(f"Failed to calculate permutation importance: {e}")
        return pd.DataFrame(columns=['Feature', 'Importance_Mean', 'Importance_Std'])

# Example usage (for testing this module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulate data and a dummy model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=3, random_state=42)
    X_dummy_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(10)])
    y_dummy_series = pd.Series(y_dummy)
    
    X_train, X_test, y_train, y_test = train_test_split(X_dummy_df, y_dummy_series, test_size=0.3, random_state=42)
    
    # Train a dummy model
    dummy_model = RandomForestClassifier(random_state=42)
    dummy_model.fit(X_train, y_train)
    
    print("\n--- Testing plot_shap_summary (bar) ---")
    plot_shap_summary(dummy_model, X_test.sample(min(50, len(X_test)), random_state=42), plot_type='bar', filepath="results/figures/test_shap_bar.png")

    print("\n--- Testing plot_shap_summary (beeswarm) ---")
    plot_shap_summary(dummy_model, X_test.sample(min(50, len(X_test)), random_state=42), plot_type='beeswarm', filepath="results/figures/test_shap_beeswarm.png")

    print("\n--- Testing calculate_permutation_importance ---")
    perm_importance_df = calculate_permutation_importance(dummy_model, X_test, y_test, metric='accuracy')
    print(perm_importance_df)
