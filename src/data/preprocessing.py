"""
Data preprocessing utilities for the healthcare analytics project.

This module provides functions for cleaning, transforming, and preparing
the healthcare length of stay dataset for model training.

@module preprocessing
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
from typing import Any, Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

def apply_age_ordinal_mapping(X: pd.DataFrame) -> pd.DataFrame:
    """
    Applies ordinal mapping to the 'Age' column and drops the original 'Age' column.
    
    Args:
        X (pd.DataFrame): Input features DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with 'Age_Ordinal' and without original 'Age'.
    """
    if 'Age' in X.columns:
        age_mapping = {
            '0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4,
            '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9,
            'More than 100 Days': 10
        }
        X_copy = X.copy()
        X_copy['Age_Ordinal'] = X_copy['Age'].map(age_mapping)
        X_copy = X_copy.drop('Age', axis=1)
        logger.info("Applied ordinal mapping to 'Age' column.")
        return X_copy
    return X

def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numeric and categorical features from the DataFrame.
    
    Args:
        X (pd.DataFrame): Input features DataFrame.
        
    Returns:
        Tuple[List[str], List[str]]: Lists of numeric and categorical feature names.
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")
    return numeric_features, categorical_features

def get_preprocessor(numeric_features: List[str], categorical_features: List[str], config: Dict[str, Any]) -> ColumnTransformer:
    """
    Creates and returns a ColumnTransformer for preprocessing.
    
    Args:
        numeric_features (List[str]): List of numeric feature names.
        categorical_features (List[str]): List of categorical feature names.
        config (Dict[str, Any]): Configuration dictionary for preprocessing steps.
                                  Expected keys: 'imputer_strategy_numeric', 'imputer_strategy_categorical'.
                                  
    Returns:
        ColumnTransformer: Configured preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get('imputer_strategy_numeric', 'median'))),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get('imputer_strategy_categorical', 'most_frequent'))),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )
    logger.info("Created ColumnTransformer for preprocessing.")
    return preprocessor

def preprocess_data_for_model(X: pd.DataFrame, y: Optional[pd.Series] = None, config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray], ColumnTransformer, LabelEncoder]:
    """
    Applies full preprocessing pipeline including age mapping, feature identification,
    and fitting/transforming data. Also handles target encoding.
    
    Args:
        X (pd.DataFrame): Input features DataFrame.
        y (pd.Series, optional): Target Series. Defaults to None.
        config (Dict[str, Any], optional): Configuration for preprocessing. Defaults to None.
                                           If None, a default config will be used.
                                           
    Returns:
        Tuple[pd.DataFrame, Optional[np.ndarray], ColumnTransformer, LabelEncoder]:
            Processed features, encoded target (if y provided), fitted preprocessor, fitted label encoder.
    """
    if config is None:
        config = {
            'imputer_strategy_numeric': 'median',
            'imputer_strategy_categorical': 'most_frequent'
        }

    X_processed_age = apply_age_ordinal_mapping(X)
    numeric_features, categorical_features = get_feature_types(X_processed_age)
    
    preprocessor = get_preprocessor(numeric_features, categorical_features, config)
    
    # Fit and transform X
    X_transformed = preprocessor.fit_transform(X_processed_age)
    
    # Convert to dense array if sparse
    if not isinstance(X_transformed, np.ndarray):
        X_transformed = X_transformed.toarray()
    
    y_encoded = None
    label_encoder = LabelEncoder()
    if y is not None:
        y_encoded = label_encoder.fit_transform(y)
        logger.info(f"Target encoded. Classes: {label_encoder.classes_}")
    
    logger.info(f"Data preprocessing complete. Transformed X shape: {X_transformed.shape}")
    return pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out()), y_encoded, preprocessor, label_encoder

def inverse_transform_predictions(y_pred_encoded: np.ndarray, label_encoder: LabelEncoder) -> pd.Series:
    """
    Inverse transforms encoded predictions back to original labels.
    
    Args:
        y_pred_encoded (np.ndarray): Encoded predictions.
        label_encoder (LabelEncoder): Fitted LabelEncoder.
        
    Returns:
        pd.Series: Original labels.
    """
    return pd.Series(label_encoder.inverse_transform(y_pred_encoded.astype(int)))

# Example usage (for testing this module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulate raw data
    data = {
        'feature_num_1': np.random.rand(100),
        'feature_num_2': np.random.randint(0, 100, 100),
        'feature_cat_1': np.random.choice(['A', 'B', 'C'], 100),
        'feature_cat_2': np.random.choice(['X', 'Y'], 100),
        'Age': np.random.choice(['0-10', '21-30', '41-50', '61-70', '81-90'], 100),
        'Stay': np.random.randint(0, 11, 100) # Target variable
    }
    df = pd.DataFrame(data)
    
    X_raw = df.drop(columns=['Stay'])
    y_raw = df['Stay']
    
    print("\n--- Testing preprocess_data_for_model ---")
    X_transformed_df, y_encoded_arr, preprocessor_obj, le_obj = preprocess_data_for_model(X_raw, y_raw)
    
    print("\nTransformed X head:")
    print(X_transformed_df.head())
    print("\nEncoded y head:")
    print(y_encoded_arr[:5])
    print("\nPreprocessor fitted features:")
    print(preprocessor_obj.get_feature_names_out())
    print("\nLabel Encoder classes:")
    print(le_obj.classes_)
    
    # Test inverse transform
    y_pred_dummy_encoded = np.random.randint(0, le_obj.classes_.shape[0], 5)
    y_pred_original = inverse_transform_predictions(y_pred_dummy_encoded, le_obj)
    print("\nInverse transformed predictions (dummy):")
    print(y_pred_original)
