"""
Cross-validation strategies for the healthcare analytics project.

This module provides functions for generating cross-validation splits
to evaluate model performance robustly.

@module cross_validation
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
import logging
from typing import Any, Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

def create_stratified_kfold_splits(
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    n_splits: int = 5, 
    shuffle: bool = True, 
    random_state: Optional[int] = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates Stratified K-Fold cross-validation splits.
    
    Args:
        X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
        y (Union[pd.Series, np.ndarray]): Target variable.
        n_splits (int): Number of folds. Defaults to 5.
        shuffle (bool): Whether to shuffle the data before splitting. Defaults to True.
        random_state (Optional[int]): Seed for reproducibility. Defaults to 42.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of (train_indices, test_indices) tuples for each fold.
        
    @example
    # Create 5-fold stratified splits
    folds = create_stratified_kfold_splits(X_features, y_labels, n_splits=5)
    for fold_num, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
        y_train, y_test = y_labels.iloc[train_idx], y_labels.iloc[test_idx]
        # ... train and evaluate model on this fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(skf.split(X, y))
    logger.info(f"Created {n_splits} stratified k-fold splits.")
    return splits

def create_kfold_splits(
    X: Union[pd.DataFrame, np.ndarray], 
    n_splits: int = 5, 
    shuffle: bool = True, 
    random_state: Optional[int] = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates K-Fold cross-validation splits (not stratified).
    
    Args:
        X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
        n_splits (int): Number of folds. Defaults to 5.
        shuffle (bool): Whether to shuffle the data before splitting. Defaults to True.
        random_state (Optional[int]): Seed for reproducibility. Defaults to 42.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of (train_indices, test_indices) tuples for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(kf.split(X))
    logger.info(f"Created {n_splits} k-fold splits.")
    return splits

def create_group_kfold_splits(
    X: Union[pd.DataFrame, np.ndarray], 
    groups: Union[pd.Series, np.ndarray],
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates Group K-Fold cross-validation splits.
    Ensures that the same group is not present in both training and testing sets for any fold.
    
    Args:
        X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
        groups (Union[pd.Series, np.ndarray]): Group labels for each sample.
        n_splits (int): Number of folds. Defaults to 5.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of (train_indices, test_indices) tuples for each fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, groups=groups)) # Pass y=None as it's not used by GroupKFold directly for splitting
    logger.info(f"Created {n_splits} group k-fold splits.")
    return splits

# Example usage (for testing this module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulate data
    X_data = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    y_data_classification = pd.Series(np.random.choice([0, 1, 2], 100)) # For stratified
    y_data_regression = pd.Series(np.random.rand(100) * 10) # For KFold (y not used for splitting but good practice)
    group_data = pd.Series(np.random.choice(['groupA', 'groupB', 'groupC', 'groupD', 'groupE'], 100))

    print("\n--- Testing create_stratified_kfold_splits ---")
    stratified_folds = create_stratified_kfold_splits(X_data, y_data_classification, n_splits=3)
    for i, (train, test) in enumerate(stratified_folds):
        logger.info(f"Fold {i+1}: Train size={len(train)}, Test size={len(test)}")
        logger.info(f"  Train target distribution: {y_data_classification.iloc[train].value_counts(normalize=True).to_dict()}")
        logger.info(f"  Test target distribution: {y_data_classification.iloc[test].value_counts(normalize=True).to_dict()}")

    print("\n--- Testing create_kfold_splits ---")
    kfold_folds = create_kfold_splits(X_data, n_splits=3)
    for i, (train, test) in enumerate(kfold_folds):
        logger.info(f"Fold {i+1}: Train size={len(train)}, Test size={len(test)}")

    print("\n--- Testing create_group_kfold_splits ---")
    group_kfold_folds = create_group_kfold_splits(X_data, groups=group_data, n_splits=3)
    for i, (train, test) in enumerate(group_kfold_folds):
        logger.info(f"Fold {i+1}: Train size={len(train)}, Test size={len(test)}")
        train_groups = set(group_data.iloc[train])
        test_groups = set(group_data.iloc[test])
        logger.info(f"  Train groups: {train_groups}")
        logger.info(f"  Test groups: {test_groups}")
        logger.info(f"  Intersection of groups: {train_groups.intersection(test_groups)}") # Should be empty
