"""
Tests for data processing modules.

@file test_data_processing.py
@version 1.0.0
@public
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.make_dataset import (
    load_raw_data, load_processed_data, validate_data_quality,
    get_target_distribution, create_sample_data
)


class TestDataLoading:
    """
    Test suite for data loading functionality.
    
    @class TestDataLoading
    @version 1.0.0
    @public
    """
    
    @pytest.mark.unit
    def test_load_raw_data_success(self, sample_data_files, test_config):
        """
        Test successful loading of raw data files.
        
        @param sample_data_files - Fixture providing sample data files
        @param test_config - Test configuration fixture
        @version 1.0.0
        @public
        """
        # Update config to point to test files
        config = test_config.copy()
        config['data']['train_path'] = str(sample_data_files['train'])
        config['data']['test_path'] = str(sample_data_files['test'])
        
        train_df, test_df = load_raw_data(config)
        
        # Assertions
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert train_df.shape[0] > 0
        assert test_df.shape[0] > 0
        assert 'lengthofstay' in train_df.columns
        assert 'lengthofstay' in test_df.columns
    
    @pytest.mark.unit
    def test_load_raw_data_missing_file(self, test_config):
        """
        Test handling of missing data files.
        
        @param test_config - Test configuration fixture
        @version 1.0.0
        @public
        """
        config = test_config.copy()
        config['data']['train_path'] = 'nonexistent_file.csv'
        
        with pytest.raises(FileNotFoundError):
            load_raw_data(config)
    
    @pytest.mark.unit
    def test_load_processed_data_success(self, sample_processed_data, temp_workspace, test_config):
        """
        Test successful loading of processed data.
        
        @param sample_processed_data - Fixture providing processed data
        @param temp_workspace - Temporary workspace fixture
        @param test_config - Test configuration fixture
        @version 1.0.0
        @public
        """
        X, y = sample_processed_data
        
        # Save processed data to temp workspace
        processed_dir = temp_workspace / 'data' / 'processed'
        X.to_csv(processed_dir / 'X_preprocessed.csv', index=False)
        y.to_csv(processed_dir / 'y_preprocessed.csv', index=False)
        
        # Update config
        config = test_config.copy()
        config['data']['processed_path'] = str(processed_dir)
        
        X_loaded, y_loaded = load_processed_data(config)
        
        # Assertions
        assert isinstance(X_loaded, pd.DataFrame)
        assert isinstance(y_loaded, pd.Series)
        assert X_loaded.shape == X.shape
        assert len(y_loaded) == len(y)
    
    @pytest.mark.unit
    def test_validate_data_quality(self, sample_raw_data):
        """
        Test data quality validation.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        train_df, _ = sample_raw_data
        
        # Test with clean data
        quality_report = validate_data_quality(train_df)
        
        assert 'missing_values' in quality_report
        assert 'duplicates' in quality_report
        assert 'data_types' in quality_report
        assert quality_report['total_rows'] == len(train_df)
        assert quality_report['total_columns'] == len(train_df.columns)
    
    @pytest.mark.unit
    def test_validate_data_quality_with_issues(self, sample_raw_data):
        """
        Test data quality validation with data issues.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        train_df, _ = sample_raw_data
        
        # Introduce data quality issues
        df_with_issues = train_df.copy()
        df_with_issues.loc[0:10, 'age'] = np.nan  # Missing values
        df_with_issues = pd.concat([df_with_issues, df_with_issues.iloc[:5]])  # Duplicates
        
        quality_report = validate_data_quality(df_with_issues)
        
        assert quality_report['missing_values']['age'] == 11
        assert quality_report['duplicates'] == 5
    
    @pytest.mark.unit
    def test_get_target_distribution(self, sample_raw_data):
        """
        Test target variable distribution analysis.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        train_df, _ = sample_raw_data
        
        distribution = get_target_distribution(train_df['lengthofstay'])
        
        assert isinstance(distribution, dict)
        assert 'value_counts' in distribution
        assert 'percentage' in distribution
        assert 'statistics' in distribution
        assert len(distribution['value_counts']) > 0
    
    @pytest.mark.unit
    def test_create_sample_data(self, sample_raw_data):
        """
        Test creation of sample datasets.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        train_df, _ = sample_raw_data
        
        # Create 10% sample
        sample_df = create_sample_data(train_df, sample_rate=0.1, random_state=42)
        
        assert isinstance(sample_df, pd.DataFrame)
        assert len(sample_df) == int(len(train_df) * 0.1)
        assert list(sample_df.columns) == list(train_df.columns)
    
    @pytest.mark.unit
    def test_create_sample_data_stratified(self, sample_raw_data):
        """
        Test creation of stratified sample datasets.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        train_df, _ = sample_raw_data
        
        # Create stratified sample
        sample_df = create_sample_data(
            train_df, 
            sample_rate=0.2, 
            stratify_column='lengthofstay',
            random_state=42
        )
        
        assert isinstance(sample_df, pd.DataFrame)
        assert len(sample_df) == int(len(train_df) * 0.2)
        
        # Check that stratification maintained class proportions
        original_props = train_df['lengthofstay'].value_counts(normalize=True)
        sample_props = sample_df['lengthofstay'].value_counts(normalize=True)
        
        # Proportions should be similar (within 5% tolerance)
        for class_label in original_props.index:
            if class_label in sample_props.index:
                assert abs(original_props[class_label] - sample_props[class_label]) < 0.05


class TestDataPreprocessing:
    """
    Test suite for data preprocessing functionality.
    
    @class TestDataPreprocessing
    @version 1.0.0
    @public
    """
    
    @pytest.mark.unit
    def test_preprocess_features_basic(self, sample_raw_data, test_config):
        """
        Test basic feature preprocessing.
        
        @param sample_raw_data - Fixture providing sample raw data
        @param test_config - Test configuration fixture
        @version 1.0.0
        @public
        """
        # Import here to avoid circular imports in testing
        from src.data.preprocessing import preprocess_features
        
        train_df, _ = sample_raw_data
        X = train_df.drop(columns=['lengthofstay'])
        
        X_processed, steps = preprocess_features(X, test_config)
        
        assert isinstance(X_processed, pd.DataFrame)
        assert isinstance(steps, list)
        assert X_processed.shape[0] == X.shape[0]
        assert len(steps) > 0
    
    @pytest.mark.unit
    def test_split_data(self, sample_processed_data, test_config):
        """
        Test data splitting functionality.
        
        @param sample_processed_data - Fixture providing processed data
        @param test_config - Test configuration fixture
        @version 1.0.0
        @public
        """
        from src.data.preprocessing import split_data
        
        X, y = sample_processed_data
        
        splits = split_data(X, y, test_config)
        
        # Check that all required splits are present
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test']
        for key in required_keys:
            assert key in splits
            assert len(splits[key]) > 0
        
        # Check proportions
        total_samples = len(X)
        test_size = test_config['data']['test_size']
        
        expected_test_size = int(total_samples * test_size)
        actual_test_size = len(splits['X_test'])
        
        # Allow for small rounding differences
        assert abs(expected_test_size - actual_test_size) <= 2
    
    @pytest.mark.unit
    def test_split_data_with_validation(self, sample_processed_data, test_config):
        """
        Test data splitting with validation set.
        
        @param sample_processed_data - Fixture providing processed data
        @param test_config - Test configuration fixture
        @version 1.0.0
        @public
        """
        from src.data.preprocessing import split_data
        
        X, y = sample_processed_data
        
        # Add validation split to config
        config_with_val = test_config.copy()
        config_with_val['data']['val_size'] = 0.2
        
        splits = split_data(X, y, config_with_val)
        
        # Check that validation splits are present
        validation_keys = ['X_val', 'y_val']
        for key in validation_keys:
            assert key in splits
            assert len(splits[key]) > 0
    
    @pytest.mark.unit
    def test_feature_scaling(self, sample_raw_data):
        """
        Test feature scaling functionality.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        from src.data.preprocessing import scale_features
        
        train_df, _ = sample_raw_data
        X = train_df.drop(columns=['lengthofstay']).select_dtypes(include=[np.number])
        
        X_scaled, scaler = scale_features(X, method='standard')
        
        assert isinstance(X_scaled, pd.DataFrame)
        assert X_scaled.shape == X.shape
        assert scaler is not None
        
        # Check that scaling worked (mean ~ 0, std ~ 1 for standard scaling)
        means = X_scaled.mean()
        stds = X_scaled.std()
        
        assert all(abs(mean) < 0.1 for mean in means)  # Close to 0
        assert all(abs(std - 1) < 0.1 for std in stds)  # Close to 1
    
    @pytest.mark.unit
    def test_encode_categorical_features(self, sample_raw_data):
        """
        Test categorical feature encoding.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        from src.data.preprocessing import encode_categorical_features
        
        train_df, _ = sample_raw_data
        
        # Create some categorical columns
        categorical_data = pd.DataFrame({
            'category_a': ['A', 'B', 'C', 'A', 'B'],
            'category_b': ['X', 'Y', 'X', 'Z', 'Y'],
            'numeric': [1, 2, 3, 4, 5]
        })
        
        encoded_data, encoders = encode_categorical_features(
            categorical_data, 
            categorical_columns=['category_a', 'category_b'],
            method='onehot'
        )
        
        assert isinstance(encoded_data, pd.DataFrame)
        assert encoded_data.shape[0] == categorical_data.shape[0]
        assert encoded_data.shape[1] > categorical_data.shape[1]  # More columns due to one-hot
        assert 'numeric' in encoded_data.columns  # Numeric column preserved
    
    @pytest.mark.unit
    def test_handle_missing_values(self, sample_raw_data):
        """
        Test missing value handling.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        from src.data.preprocessing import handle_missing_values
        
        train_df, _ = sample_raw_data
        
        # Introduce missing values
        df_with_missing = train_df.copy()
        df_with_missing.loc[0:10, 'age'] = np.nan
        df_with_missing.loc[5:15, 'num_procedures'] = np.nan
        
        df_imputed, imputers = handle_missing_values(
            df_with_missing,
            strategy='mean'
        )
        
        assert isinstance(df_imputed, pd.DataFrame)
        assert df_imputed.shape == df_with_missing.shape
        assert df_imputed.isnull().sum().sum() == 0  # No missing values left
    
    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_raw_data, test_config):
        """
        Test the complete preprocessing pipeline.
        
        @param sample_raw_data - Fixture providing sample raw data
        @param test_config - Test configuration fixture
        @version 1.0.0
        @public
        """
        from src.data.preprocessing import preprocess_features
        
        train_df, _ = sample_raw_data
        X = train_df.drop(columns=['lengthofstay'])
        
        # Add some missing values and categorical data to make it realistic
        X_with_issues = X.copy()
        X_with_issues.loc[0:5, 'age'] = np.nan
        X_with_issues['category_feature'] = ['A', 'B', 'C'] * (len(X_with_issues) // 3 + 1)
        X_with_issues = X_with_issues.iloc[:len(X)]  # Trim to original size
        
        X_processed, steps = preprocess_features(X_with_issues, test_config)
        
        # Comprehensive checks
        assert isinstance(X_processed, pd.DataFrame)
        assert X_processed.shape[0] == X_with_issues.shape[0]
        assert X_processed.isnull().sum().sum() == 0  # No missing values
        assert all(isinstance(x, (int, float, np.integer, np.floating)) 
                  for col in X_processed.columns 
                  for x in X_processed[col])  # All numeric
        assert len(steps) > 0


class TestDataValidation:
    """
    Test suite for data validation functionality.
    
    @class TestDataValidation
    @version 1.0.0
    @public
    """
    
    @pytest.mark.unit
    def test_schema_validation(self, sample_raw_data):
        """
        Test data schema validation.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        from src.data.validation import validate_schema
        
        train_df, _ = sample_raw_data
        
        # Define expected schema
        expected_schema = {
            'age': 'int64',
            'gender': 'int64',
            'lengthofstay': 'int64'
        }
        
        is_valid, errors = validate_schema(train_df, expected_schema)
        
        if not is_valid:
            # Should contain helpful error messages
            assert isinstance(errors, list)
            assert len(errors) > 0
    
    @pytest.mark.unit
    def test_data_drift_detection(self, sample_raw_data):
        """
        Test data drift detection.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        from src.data.validation import detect_data_drift
        
        train_df, test_df = sample_raw_data
        
        # Test with same distribution (should have low drift)
        drift_report = detect_data_drift(
            train_df.drop(columns=['lengthofstay']),
            train_df.drop(columns=['lengthofstay'])  # Same data
        )
        
        assert isinstance(drift_report, dict)
        assert 'overall_drift_score' in drift_report
        assert 'feature_drift_scores' in drift_report
        assert drift_report['overall_drift_score'] < 0.1  # Low drift expected
    
    @pytest.mark.unit
    def test_outlier_detection(self, sample_raw_data):
        """
        Test outlier detection functionality.
        
        @param sample_raw_data - Fixture providing sample raw data
        @version 1.0.0
        @public
        """
        from src.data.validation import detect_outliers
        
        train_df, _ = sample_raw_data
        
        outliers = detect_outliers(
            train_df.select_dtypes(include=[np.number]),
            method='isolation_forest'
        )
        
        assert isinstance(outliers, pd.Series)
        assert len(outliers) == len(train_df)
        assert outliers.dtype == bool 