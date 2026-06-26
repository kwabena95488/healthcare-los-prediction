"""
Data validation utilities for the healthcare analytics project.

This module provides functions for validating data integrity, schema,
and quality.

@module validation
@version 1.0.0
@public
"""

import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, type]) -> Tuple[bool, List[str]]:
    """
    Validates the DataFrame schema against an expected schema.
    Checks for presence of columns and their data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        expected_schema (Dict[str, type]): A dictionary where keys are column names
                                           and values are expected Python types (e.g., int, float, str, object).
                                           
    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating if the schema is valid,
                                and a list of error messages if invalid.
                                
    @example
    schema = {'col_a': int, 'col_b': str}
    is_valid, errors = validate_schema(my_df, schema)
    if not is_valid:
        for error in errors:
            print(error)
    """
    errors = []
    is_valid = True

    # Check for missing columns
    for col_name in expected_schema.keys():
        if col_name not in df.columns:
            errors.append(f"Missing expected column: '{col_name}'")
            is_valid = False
    
    # Check for unexpected columns (optional, can be strict)
    # for col_name in df.columns:
    #     if col_name not in expected_schema.keys():
    #         errors.append(f"Unexpected column found: '{col_name}'")
    #         is_valid = False

    if not is_valid: # Stop if columns are missing, as type checks might fail
        return is_valid, errors

    # Check data types
    for col_name, expected_type in expected_schema.items():
        actual_type = df[col_name].dtype
        # Basic type mapping; can be expanded
        if expected_type == int and not pd.api.types.is_integer_dtype(actual_type):
            errors.append(f"Column '{col_name}': Expected type {expected_type}, found {actual_type}")
            is_valid = False
        elif expected_type == float and not pd.api.types.is_float_dtype(actual_type):
            errors.append(f"Column '{col_name}': Expected type {expected_type}, found {actual_type}")
            is_valid = False
        elif expected_type == str and not (pd.api.types.is_string_dtype(actual_type) or pd.api.types.is_object_dtype(actual_type)):
            errors.append(f"Column '{col_name}': Expected type {expected_type}, found {actual_type}")
            is_valid = False
        elif expected_type == object and not pd.api.types.is_object_dtype(actual_type): # For general objects or mixed types
             errors.append(f"Column '{col_name}': Expected type {expected_type}, found {actual_type}")
             is_valid = False
        # Add more type checks as needed (e.g., bool, datetime)

    if is_valid:
        logger.info("Schema validation passed.")
    else:
        logger.warning(f"Schema validation failed with {len(errors)} errors.")
        for error in errors:
            logger.warning(f"  - {error}")
            
    return is_valid, errors

def check_missing_values(df: pd.DataFrame, threshold_map: Optional[Dict[str, float]] = None) -> Tuple[bool, Dict[str, float]]:
    """
    Checks for missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to check.
        threshold_map (Optional[Dict[str, float]]): A dictionary mapping column names to
                                                    their allowed missing value percentage threshold (0.0 to 1.0).
                                                    If None, just reports missing values.
                                                    
    Returns:
        Tuple[bool, Dict[str, float]]: A tuple containing a boolean indicating if thresholds are met (if provided),
                                       and a dictionary of columns with their missing value percentages.
    """
    missing_percentages = df.isnull().mean()
    missing_info = {col: perc for col, perc in missing_percentages.items() if perc > 0}
    
    all_thresholds_met = True
    if threshold_map:
        for col, threshold in threshold_map.items():
            if col in missing_percentages and missing_percentages[col] > threshold:
                logger.warning(f"Column '{col}' missing values ({missing_percentages[col]:.2%}) exceed threshold ({threshold:.2%}).")
                all_thresholds_met = False
    
    if not missing_info:
        logger.info("No missing values found.")
    else:
        logger.info("Missing values found:")
        for col, perc in missing_info.items():
            logger.info(f"  - Column '{col}': {df[col].isnull().sum()} missing ({perc:.2%})")
            
    return all_thresholds_met if threshold_map else True, missing_info

def check_duplicate_rows(df: pd.DataFrame) -> Tuple[bool, int]:
    """
    Checks for duplicate rows in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to check.
        
    Returns:
        Tuple[bool, int]: A tuple containing a boolean (True if no duplicates) and the count of duplicate rows.
    """
    num_duplicates = df.duplicated().sum()
    if num_duplicates == 0:
        logger.info("No duplicate rows found.")
        return True, 0
    else:
        logger.warning(f"Found {num_duplicates} duplicate rows.")
        return False, num_duplicates

def check_numerical_range(df: pd.DataFrame, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> bool:
    """
    Checks if values in a numerical column are within a specified range.
    
    Args:
        df (pd.DataFrame): DataFrame to check.
        column (str): Name of the numerical column.
        min_val (Optional[float]): Minimum allowed value. Defaults to None.
        max_val (Optional[float]): Maximum allowed value. Defaults to None.
        
    Returns:
        bool: True if all values are within range, False otherwise.
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found for range check.")
        return False
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.error(f"Column '{column}' is not numeric, cannot perform range check.")
        return False

    is_valid = True
    if min_val is not None:
        if (df[column] < min_val).any():
            logger.warning(f"Column '{column}' has values below minimum {min_val}.")
            is_valid = False
    if max_val is not None:
        if (df[column] > max_val).any():
            logger.warning(f"Column '{column}' has values above maximum {max_val}.")
            is_valid = False
    
    if is_valid:
        logger.info(f"Range check passed for column '{column}'.")
    return is_valid

def check_categorical_values(df: pd.DataFrame, column: str, allowed_values: List[Any]) -> bool:
    """
    Checks if values in a categorical column are within a specified set of allowed values.
    
    Args:
        df (pd.DataFrame): DataFrame to check.
        column (str): Name of the categorical column.
        allowed_values (List[Any]): List of allowed values.
        
    Returns:
        bool: True if all values are allowed, False otherwise.
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found for value check.")
        return False
        
    unexpected_values = df[~df[column].isin(allowed_values)][column].unique()
    if len(unexpected_values) == 0:
        logger.info(f"Categorical value check passed for column '{column}'.")
        return True
    else:
        logger.warning(f"Column '{column}' contains unexpected values: {list(unexpected_values)}.")
        return False

# Example usage (for testing this module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulate data
    data = {
        'id': range(10),
        'age': [25, 30, np.nan, 40, 25, 50, 55, 60, 65, 70],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'O'], # 'O' is unexpected
        'score': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    }
    test_df = pd.DataFrame(data)
    test_df.iloc[5] = test_df.iloc[4] # Create a duplicate row
    
    print("\n--- Testing validate_schema ---")
    expected_schema = {'id': int, 'age': float, 'gender': str, 'score': float, 'extra_col': str}
    validate_schema(test_df, expected_schema)
    
    expected_schema_correct = {'id': int, 'age': float, 'gender': object, 'score': float}
    validate_schema(test_df, expected_schema_correct)

    print("\n--- Testing check_missing_values ---")
    check_missing_values(test_df, threshold_map={'age': 0.05}) # age missing > 5%
    check_missing_values(test_df, threshold_map={'age': 0.15}) # age missing <= 15%

    print("\n--- Testing check_duplicate_rows ---")
    check_duplicate_rows(test_df)

    print("\n--- Testing check_numerical_range ---")
    check_numerical_range(test_df, 'age', min_val=0, max_val=100)
    check_numerical_range(test_df, 'score', min_val=0, max_val=5.0) # Will fail

    print("\n--- Testing check_categorical_values ---")
    check_categorical_values(test_df, 'gender', ['M', 'F']) # Will fail due to 'O'
    check_categorical_values(test_df, 'gender', ['M', 'F', 'O'])
