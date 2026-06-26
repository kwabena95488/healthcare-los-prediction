"""
Model comparison utilities for the healthcare analytics project.

This module provides functions for comparing the performance of different models
based on their evaluation metrics, typically from MLflow runs.

@module model_comparison
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
import mlflow
from mlflow.entities import ViewType
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

def get_mlflow_run_metrics(experiment_name: str, run_ids: Optional[List[str]] = None, metric_prefix: str = "avg_") -> Dict[str, Dict[str, float]]:
    """
    Retrieves average metrics for specified MLflow runs within an experiment.
    
    Args:
        experiment_name (str): Name of the MLflow experiment.
        run_ids (Optional[List[str]]): List of specific run IDs to retrieve metrics for.
                                       If None, retrieves metrics for all runs in the experiment.
        metric_prefix (str): Prefix for the metrics to retrieve (e.g., "avg_" for average metrics).
                              
    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are run names (or run IDs if name is missing)
                                     and values are dictionaries of their metrics.
                                     
    @example
    # Get metrics for all runs in an experiment
    all_model_metrics = get_mlflow_run_metrics("Healthcare_LOS_Prediction")
    
    # Get metrics for specific runs
    selected_model_metrics = get_mlflow_run_metrics("Healthcare_LOS_Prediction", run_ids=['run1_id', 'run2_id'])
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        logger.error(f"MLflow experiment '{experiment_name}' not found.")
        return {}

    runs_data = {}
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1000, # Adjust as needed
        order_by=["attribute.start_time DESC"]
    )

    for run in runs:
        if run_ids is None or run.info.run_id in run_ids:
            run_name = run.data.tags.get('mlflow.runName', run.info.run_id)
            metrics = {k.replace(metric_prefix, ''): v for k, v in run.data.metrics.items() if k.startswith(metric_prefix)}
            if metrics:
                runs_data[run_name] = metrics
                logger.info(f"Retrieved metrics for run '{run_name}' ({run.info.run_id}).")
            else:
                logger.warning(f"No metrics with prefix '{metric_prefix}' found for run '{run_name}'.")
    
    if not runs_data:
        logger.warning(f"No runs found or no metrics retrieved for experiment '{experiment_name}'.")

    return runs_data

def compare_models_by_metric(
    model_metrics: Dict[str, Dict[str, float]], 
    metric_name: str, 
    ascending: bool = False
) -> pd.DataFrame:
    """
    Compares models based on a specific metric.
    
    Args:
        model_metrics (Dict[str, Dict[str, float]]): Dictionary of model metrics (output of get_mlflow_run_metrics).
        metric_name (str): The name of the metric to compare (e.g., 'accuracy', 'f1_weighted').
        ascending (bool): Whether to sort in ascending order. Defaults to False (descending).
        
    Returns:
        pd.DataFrame: A DataFrame showing models ranked by the specified metric.
        
    @example
    # Compare models by accuracy
    accuracy_comparison = compare_models_by_metric(all_model_metrics, 'accuracy')
    print(accuracy_comparison)
    """
    comparison_data = []
    for model_name, metrics in model_metrics.items():
        if metric_name in metrics:
            comparison_data.append({'Model': model_name, metric_name: metrics[metric_name]})
        else:
            logger.warning(f"Metric '{metric_name}' not found for model '{model_name}'.")
            
    if not comparison_data:
        logger.warning(f"No data to compare for metric '{metric_name}'.")
        return pd.DataFrame(columns=['Model', metric_name])

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values(by=metric_name, ascending=ascending).reset_index(drop=True)
    logger.info(f"Models compared by metric '{metric_name}'.")
    return df_comparison

def plot_model_comparison(
    df_comparison: pd.DataFrame, 
    metric_name: str, 
    title: str = "Model Comparison", 
    filepath: Optional[Union[str, Path]] = None
) -> None:
    """
    Generates a bar plot comparing models based on a specific metric.
    
    Args:
        df_comparison (pd.DataFrame): DataFrame containing model comparison data (output of compare_models_by_metric).
        metric_name (str): The name of the metric being plotted.
        title (str): Title of the plot. Defaults to "Model Comparison".
        filepath (Optional[Union[str, Path]]): Path to save the plot. If None, displays the plot.
                                                
    @example
    # Plot model comparison
    plot_model_comparison(accuracy_comparison, 'accuracy', title="Model Accuracy Comparison", filepath="results/figures/model_accuracy.png")
    """
    if df_comparison.empty:
        logger.warning("No data to plot for model comparison.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric_name, data=df_comparison, palette='viridis')
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        logger.info(f"Model comparison plot saved to {filepath}")
    else:
        plt.show()
    plt.close() # Close plot to free memory

# Example usage (for testing this module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulate MLflow runs and metrics for testing
    # In a real scenario, you would have actual MLflow runs
    
    # Create a dummy MLflow experiment for testing
    mlflow.set_experiment("Test_Model_Comparison")
    
    # Simulate some runs
    with mlflow.start_run(run_name="ModelA"):
        mlflow.log_metric("avg_accuracy", 0.85)
        mlflow.log_metric("avg_f1_weighted", 0.84)
        mlflow.log_metric("avg_precision_weighted", 0.83)
    with mlflow.start_run(run_name="ModelB"):
        mlflow.log_metric("avg_accuracy", 0.88)
        mlflow.log_metric("avg_f1_weighted", 0.87)
        mlflow.log_metric("avg_precision_weighted", 0.86)
    with mlflow.start_run(run_name="ModelC"):
        mlflow.log_metric("avg_accuracy", 0.82)
        mlflow.log_metric("avg_f1_weighted", 0.81)
        mlflow.log_metric("avg_precision_weighted", 0.80)

    print("\n--- Testing get_mlflow_run_metrics ---")
    all_metrics = get_mlflow_run_metrics("Test_Model_Comparison")
    print(all_metrics)

    print("\n--- Testing compare_models_by_metric (accuracy) ---")
    accuracy_comp = compare_models_by_metric(all_metrics, 'accuracy')
    print(accuracy_comp)

    print("\n--- Testing plot_model_comparison (accuracy) ---")
    plot_model_comparison(accuracy_comp, 'accuracy', title="Accuracy Comparison", filepath="results/figures/test_accuracy_comparison.png")

    print("\n--- Testing compare_models_by_metric (f1_weighted) ---")
    f1_comp = compare_models_by_metric(all_metrics, 'f1_weighted')
    print(f1_comp)

    print("\n--- Testing plot_model_comparison (f1_weighted) ---")
    plot_model_comparison(f1_comp, 'f1_weighted', title="F1 Weighted Comparison", filepath="results/figures/test_f1_comparison.png")

    # Clean up dummy experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Test_Model_Comparison")
    if experiment:
        client.delete_experiment(experiment.experiment_id)
        logger.info("Cleaned up 'Test_Model_Comparison' experiment.")
