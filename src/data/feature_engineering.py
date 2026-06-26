"""
Feature engineering utilities for the healthcare analytics project.

This module provides functions for creating new features, transforming existing ones,
and selecting relevant features for model training.

@module feature_engineering
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Creates interaction features by multiplying pairs of specified features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_pairs (List[Tuple[str, str]]): List of tuples, where each tuple contains
                                                two feature names to interact.
                                                
    Returns:
        pd.DataFrame: DataFrame with new interaction features.
        
    @example
    # Create interaction features
    df = create_interaction_features(df, [('feat1', 'feat2'), ('feat3', 'feat4')])
    """
    df_copy = df.copy()
    for feat1, feat2 in feature_pairs:
        if feat1 in df_copy.columns and feat2 in df_copy.columns:
            new_feature_name = f"{feat1}_x_{feat2}"
            df_copy[new_feature_name] = df_copy[feat1] * df_copy[feat2]
            logger.info(f"Created interaction feature: {new_feature_name}")
        else:
            logger.warning(f"One or both features not found for interaction: {feat1}, {feat2}")
    return df_copy

def create_polynomial_features(df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Creates polynomial features for specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (List[str]): List of feature names to create polynomial features for.
        degree (int): The degree of the polynomial features. Defaults to 2.
        
    Returns:
        pd.DataFrame: DataFrame with new polynomial features.
        
    @example
    # Create polynomial features
    df = create_polynomial_features(df, ['feat1', 'feat2'], degree=3)
    """
    df_copy = df.copy()
    for feature in features:
        if feature in df_copy.columns:
            for d in range(2, degree + 1):
                new_feature_name = f"{feature}_pow{d}"
                df_copy[new_feature_name] = df_copy[feature] ** d
                logger.info(f"Created polynomial feature: {new_feature_name}")
        else:
            logger.warning(f"Feature not found for polynomial transformation: {feature}")
    return df_copy

def apply_log_transform(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Applies a log transformation (log1p) to specified features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (List[str]): List of feature names to apply log transformation to.
        
    Returns:
        pd.DataFrame: DataFrame with log-transformed features.
        
    @example
    # Apply log transform
    df = apply_log_transform(df, ['skewed_feat1', 'skewed_feat2'])
    """
    df_copy = df.copy()
    for feature in features:
        if feature in df_copy.columns:
            # Ensure the feature is numeric and non-negative before log transform
            if pd.api.types.is_numeric_dtype(df_copy[feature]) and (df_copy[feature] >= 0).all():
                df_copy[feature] = np.log1p(df_copy[feature])
                logger.info(f"Applied log transform to feature: {feature}")
            else:
                logger.warning(f"Cannot apply log transform to non-numeric or negative feature: {feature}")
        else:
            logger.warning(f"Feature not found for log transformation: {feature}")
    return df_copy

# Add more feature engineering functions as needed, e.g.:
# - binning_numerical_features
# - encoding_cyclical_features
# - feature_scaling (though often part of preprocessing)
# - dimensionality_reduction (e.g., PCA)

# Example usage (for testing this module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulate data
    data = {
        'feat1': np.random.rand(100) * 10,
        'feat2': np.random.rand(100) * 5,
        'skewed_feat1': np.random.gamma(1, 2, 100) * 100,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    test_df = pd.DataFrame(data)
    
    print("\n--- Original DataFrame ---")
    print(test_df.head())
    
    print("\n--- Testing create_interaction_features ---")
    df_interact = create_interaction_features(test_df, [('feat1', 'feat2')])
    print(df_interact.head())
    
    print("\n--- Testing create_polynomial_features ---")
    df_poly = create_polynomial_features(test_df, ['feat1'], degree=3)
    print(df_poly.head())
    
    print("\n--- Testing apply_log_transform ---")
    df_log = apply_log_transform(test_df, ['skewed_feat1'])
    print(df_log.head())
    
    print("\n--- Chained operations ---")
    df_chained = test_df.pipe(create_interaction_features, [('feat1', 'feat2')]) \
                        .pipe(create_polynomial_features, ['feat1'], degree=2) \
                        .pipe(apply_log_transform, ['skewed_feat1'])
    print(df_chained.head())
    print(f"Final shape: {df_chained.shape}")
