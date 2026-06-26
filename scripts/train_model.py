"""
Command-line interface for training models in the healthcare analytics project.

This script allows users to train different models (e.g., Logistic Regression,
XGBoost, LightGBM, CatBoost, Neural Network, Ensemble) using configurations
defined in YAML files.

@module train_model
@version 1.0.0
@public
"""

import argparse
import logging
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utilities and models
from src.utils.config import load_config, setup_logging
from src.data.make_dataset import load_raw_data
from src.evaluation.cross_validation import create_stratified_kfold_splits

# Import refactored models
from src.models.baseline_model import BaselineModel
from src.models.gradient_boosting.xgboost_model import XGBoostModel
from src.models.gradient_boosting.lightgbm_model import LightGBMModel
from src.models.gradient_boosting.catboost_model import CatBoostModel
from src.models.neural_network_model import NeuralNetworkModel
from src.models.ensemble_model import EnsembleModel

def train_model(model_name: str, config: Dict[str, Any]) -> None:
    """
    Orchestrates the training process for a specified model.
    
    Args:
        model_name (str): Name of the model to train (e.g., 'baseline', 'xgboost').
        config (Dict[str, Any]): Full configuration dictionary.
        
    @example
    # To train a baseline model using CLI:
    # python scripts/train_model.py --model baseline
    """
    logger.info(f"Starting training for model: {model_name}")

    # Setup MLflow experiment
    mlflow_config = config.get('mlflow', {})
    experiment_name = mlflow_config.get('experiment_name', 'Healthcare_LOS_Prediction')
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: {experiment_name}")

    # Load data
    try:
        train_df, _ = load_raw_data(config)
        X = train_df.drop(columns=['Stay', 'case_id'])
        y = train_df['Stay']
        logger.info(f"Raw data loaded. X shape: {X.shape}, y shape: {y.shape}")
    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}. Please ensure data files are in data/raw/ and configured correctly.")
        return

    # Get model-specific configuration
    model_config = config['models'].get(model_name, {})
    if not model_config:
        logger.error(f"Configuration for model '{model_name}' not found in config/model_config.yaml.")
        return

    # Instantiate the model
    model_instance = None
    if model_name == 'baseline':
        model_instance = BaselineModel(config=model_config)
    elif model_name == 'xgboost':
        model_instance = XGBoostModel(config=model_config)
    elif model_name == 'lightgbm':
        model_instance = LightGBMModel(config=model_config)
    elif model_name == 'catboost':
        model_instance = CatBoostModel(config=model_config)
    elif model_name == 'neural_network':
        model_instance = NeuralNetworkModel(config=model_config)
    elif model_name == 'ensemble':
        model_instance = EnsembleModel(config=model_config)
    else:
        logger.error(f"Unknown model type specified: {model_name}")
        return

    # Create cross-validation folds
    cv_config = config.get('cross_validation', {})
    n_splits = cv_config.get('n_splits', 5)
    random_state = cv_config.get('random_state', 42)
    
    try:
        cv_folds = create_stratified_kfold_splits(X, y, n_splits=n_splits, random_state=random_state)
        logger.info(f"Created {n_splits} stratified cross-validation folds.")
    except Exception as e:
        logger.error(f"Failed to create CV splits: {e}. Ensure target variable is suitable for stratification.")
        return

    all_y_test = []
    all_y_pred = []
    
    with mlflow.start_run(run_name=f"{model_name.capitalize()}_Training_Run"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model_config) # Log model-specific config
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
            logger.info(f"--- Training Fold {fold_idx+1}/{n_splits} ---")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train the model
            model_instance.train(X_train, y_train, fold_idx=fold_idx)
            
            # Make predictions
            y_pred = model_instance.predict(X_test)
            
            # Evaluate and log fold metrics
            metrics = model_instance.evaluate(y_test, y_pred, fold_idx=fold_idx)
            
            all_y_test.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
        
        # Calculate and log overall metrics
        overall_accuracy = accuracy_score(all_y_test, all_y_pred)
        overall_f1_weighted = f1_score(all_y_test, all_y_pred, average='weighted')
        overall_f1_macro = f1_score(all_y_test, all_y_pred, average='macro')
        
        mlflow.log_metric("overall_accuracy", overall_accuracy)
        mlflow.log_metric("overall_f1_weighted", overall_f1_weighted)
        mlflow.log_metric("overall_f1_macro", overall_f1_macro)
        logger.info(f"Overall Accuracy: {overall_accuracy:.4f}, F1 (weighted): {overall_f1_weighted:.4f}, F1 (macro): {overall_f1_macro:.4f}")

        # Log overall artifacts (confusion matrix, classification report)
        # Pass original y_data to evaluate for correct label encoding in plotting
        model_instance.evaluate(pd.Series(all_y_test), pd.Series(all_y_pred), log_artifacts=True)
        
        # Log the final model (the last trained model from the CV loop)
        if hasattr(model_instance, 'pipeline') and model_instance.pipeline is not None:
            mlflow.sklearn.log_model(model_instance.pipeline, f"{model_name}_model_pipeline")
        elif hasattr(model_instance, 'model') and model_instance.model is not None:
            if model_name == 'neural_network':
                mlflow.tensorflow.log_model(model_instance.model, f"{model_name}_model")
            elif model_name == 'catboost':
                mlflow.catboost.log_model(model_instance.model, f"{model_name}_model")
            elif model_name == 'xgboost':
                mlflow.xgboost.log_model(model_instance.model, f"{model_name}_model")
            elif model_name == 'lightgbm':
                mlflow.lightgbm.log_model(model_instance.model, f"{model_name}_model")
            elif model_name == 'ensemble':
                mlflow.sklearn.log_model(model_instance.ensemble_classifier, f"{model_name}_model")
            else:
                logger.warning(f"Could not log model for {model_name} using specific MLflow flavor. Logging as generic sklearn model.")
                mlflow.sklearn.log_model(model_instance.model, f"{model_name}_model")
        else:
            logger.warning(f"No scikit-learn compatible model or pipeline found for {model_name} to log.")

    logger.info(f"Training and evaluation for {model_name} completed and logged to MLflow!")

def main():
    parser = argparse.ArgumentParser(description="Train a model for Healthcare Length of Stay Prediction.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'xgboost', 'lightgbm', 'catboost', 'neural_network', 'ensemble'],
                        help='Name of the model to train.')
    parser.add_argument('--config_path', type=str, default='config/config.yaml',
                        help='Path to the main configuration YAML file.')
    
    args = parser.parse_args()

    # Load main configuration
    config = load_config(args.config_path)
    
    # Setup logging based on config
    setup_logging(config.get('logging', {}))

    # Run training
    train_model(args.model, config)

if __name__ == "__main__":
    main()
