"""
Data loading utilities for the healthcare analytics project.

This module provides functions for loading and basic preprocessing of the
healthcare length of stay dataset.

@module make_dataset
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from ..utils.config import load_config, get_data_path

logger = logging.getLogger(__name__)

def load_raw_data(config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw training and test datasets.
    
    @param {Dict[str, Any]} [config] - Configuration dictionary (optional)
    @returns {Tuple[pd.DataFrame, pd.DataFrame]} Training and test dataframes
    @throws {FileNotFoundError} When data files are not found
    @version 1.0.0
    @public
    
    @example
    # Load raw data
    train_df, test_df = load_raw_data()
    """
    if config is None:
        config = load_config()
    
    data_path = get_data_path(config)
    
    train_file = data_path / config['data']['train_file']
    test_file = data_path / config['data']['test_file']
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test data file not found: {test_file}")
    
    logger.info(f"Loading training data from {train_file}")
    train_df = pd.read_csv(train_file)
    
    logger.info(f"Loading test data from {test_file}")
    test_df = pd.read_csv(test_file)
    
    logger.info(f"Loaded training data: {train_df.shape}")
    logger.info(f"Loaded test data: {test_df.shape}")
    
    return train_df, test_df

def load_processed_data(config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load preprocessed training data.
    
    @param {Dict[str, Any]} [config] - Configuration dictionary (optional)
    @returns {Tuple[pd.DataFrame, pd.Series]} Features and target data
    @throws {FileNotFoundError} When processed data files are not found
    @version 1.0.0
    @public
    
    @example
    # Load processed data
    X, y = load_processed_data()
    """
    if config is None:
        config = load_config()
    
    processed_path = Path(config['data']['processed_data_path'])
    
    X_file = processed_path / 'X_preprocessed.csv'
    y_file = processed_path / 'y_preprocessed.csv'
    
    if not X_file.exists():
        raise FileNotFoundError(f"Processed features file not found: {X_file}")
    if not y_file.exists():
        raise FileNotFoundError(f"Processed target file not found: {y_file}")
    
    logger.info(f"Loading processed features from {X_file}")
    X = pd.read_csv(X_file)
    
    logger.info(f"Loading processed target from {y_file}")
    y = pd.read_csv(y_file).squeeze()  # Convert to Series
    
    logger.info(f"Loaded processed features: {X.shape}")
    logger.info(f"Loaded processed target: {y.shape}")
    
    return X, y

def validate_data(df: pd.DataFrame, data_type: str = "dataset") -> Dict[str, Any]:
    """
    Validate dataset and return basic statistics.
    
    @param {pd.DataFrame} df - Dataset to validate
    @param {string} [data_type="dataset"] - Type of data for logging
    @returns {Dict[str, Any]} Validation statistics
    @version 1.0.0
    @public
    
    @example
    # Validate dataset
    stats = validate_data(train_df, "training")
    """
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    logger.info(f"{data_type.capitalize()} validation:")
    logger.info(f"  Shape: {stats['shape']}")
    logger.info(f"  Missing values: {sum(stats['missing_values'].values())}")
    logger.info(f"  Duplicate rows: {stats['duplicate_rows']}")
    logger.info(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
    
    return stats

def get_target_distribution(y: pd.Series) -> Dict[str, Any]:
    """
    Get target variable distribution statistics.
    
    @param {pd.Series} y - Target variable
    @returns {Dict[str, Any]} Distribution statistics
    @version 1.0.0
    @public
    
    @example
    # Get target distribution
    target_stats = get_target_distribution(y_train)
    """
    distribution = {
        'value_counts': y.value_counts().to_dict(),
        'proportions': y.value_counts(normalize=True).to_dict(),
        'unique_values': list(y.unique()),
        'num_classes': y.nunique()
    }
    
    logger.info("Target variable distribution:")
    for value, count in distribution['value_counts'].items():
        prop = distribution['proportions'][value]
        logger.info(f"  Class {value}: {count} ({prop:.2%})")
    
    return distribution

def save_data(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    """
    Save dataframe to CSV file.
    
    @param {pd.DataFrame} df - Dataframe to save
    @param {Path} filepath - File path to save to
    @param {boolean} [index=False] - Whether to save index
    @version 1.0.0
    @public
    
    @example
    # Save processed data
    save_data(processed_df, Path('data/processed/features.csv'))
    """
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving data to {filepath}")
    df.to_csv(filepath, index=index)
    logger.info(f"Data saved successfully: {df.shape}")

def load_data_sample(n_samples: int = 1000, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load a sample of the training data for quick testing.
    
    @param {number} [n_samples=1000] - Number of samples to load
    @param {Dict[str, Any]} [config] - Configuration dictionary (optional)
    @returns {pd.DataFrame} Sample of training data
    @version 1.0.0
    @public
    
    @example
    # Load sample data for quick testing
    sample_df = load_data_sample(500)
    """
    train_df, _ = load_raw_data(config)
    
    if len(train_df) <= n_samples:
        logger.warning(f"Dataset has only {len(train_df)} rows, returning full dataset")
        return train_df
    
    sample = train_df.sample(n=n_samples, random_state=42)
    logger.info(f"Loaded sample data: {sample.shape}")
    
    return sample 