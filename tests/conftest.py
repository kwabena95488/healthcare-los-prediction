"""
Pytest configuration and shared fixtures for Healthcare Length of Stay tests.

@file conftest.py
@version 1.0.0
@public
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """
    Test configuration fixture.
    
    @returns {Dict[str, Any]} Test configuration
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_something(test_config):
        assert test_config['project']['name'] == 'healthcare-los-test'
    """
    return {
        'project': {
            'name': 'healthcare-los-test',
            'version': '1.0.0-test'
        },
        'data': {
            'raw_path': 'tests/fixtures/data/raw',
            'processed_path': 'tests/fixtures/data/processed',
            'target_column': 'lengthofstay',
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42
        },
        'models': {
            'output_path': 'tests/fixtures/models',
            'random_state': 42
        },
        'results': {
            'output_path': 'tests/fixtures/results'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }


@pytest.fixture(scope="session")
def temp_workspace():
    """
    Create a temporary workspace for tests.
    
    @returns {Path} Temporary workspace directory
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_file_operations(temp_workspace):
        test_file = temp_workspace / 'test.txt'
        test_file.write_text('test content')
    """
    temp_dir = tempfile.mkdtemp(prefix="healthcare_test_")
    temp_path = Path(temp_dir)
    
    # Create directory structure
    (temp_path / 'data' / 'raw').mkdir(parents=True)
    (temp_path / 'data' / 'processed').mkdir(parents=True)
    (temp_path / 'data' / 'interim').mkdir(parents=True)
    (temp_path / 'models').mkdir(parents=True)
    (temp_path / 'results' / 'figures').mkdir(parents=True)
    (temp_path / 'results' / 'metrics').mkdir(parents=True)
    (temp_path / 'results' / 'reports').mkdir(parents=True)
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sample raw training and test data.
    
    @returns {Tuple[pd.DataFrame, pd.DataFrame]} Training and test dataframes
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_data_loading(sample_raw_data):
        train_df, test_df = sample_raw_data
        assert train_df.shape[0] > 0
    """
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,  # 4 length of stay categories
        random_state=42
    )
    
    # Create feature names
    feature_names = [
        'age', 'gender', 'admission_type', 'severity_illness', 
        'risk_mortality', 'bed_census', 'num_procedures',
        'num_medications', 'primary_diagnosis', 'secondary_diagnosis',
        'apr_drg_code', 'apr_medical_surgical', 'hospital_county',
        'insurance_type', 'race', 'ethnicity', 'previous_admissions',
        'comorbidity_score', 'lab_value_1', 'lab_value_2'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['lengthofstay'] = y
    
    # Add some realistic data transformations
    df['age'] = (df['age'] * 30 + 65).astype(int)  # Age 35-95
    df['gender'] = (df['gender'] > 0).astype(int)  # Binary gender
    df['admission_type'] = (df['admission_type'] * 3).astype(int) % 4  # 4 admission types
    
    # Split into train and test
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


@pytest.fixture
def sample_processed_data(sample_raw_data) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate sample processed data.
    
    @returns {Tuple[pd.DataFrame, pd.Series]} Processed features and target
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_preprocessing(sample_processed_data):
        X, y = sample_processed_data
        assert X.shape[1] > 0
    """
    train_df, _ = sample_raw_data
    
    # Simple preprocessing
    X = train_df.drop(columns=['lengthofstay'])
    y = train_df['lengthofstay']
    
    # Scale features to [0, 1]
    X = (X - X.min()) / (X.max() - X.min())
    
    return X, y


@pytest.fixture
def trained_model(sample_processed_data) -> RandomForestClassifier:
    """
    Generate a trained model for testing.
    
    @returns {RandomForestClassifier} Trained model
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_model_prediction(trained_model, sample_processed_data):
        X, y = sample_processed_data
        predictions = trained_model.predict(X)
        assert len(predictions) == len(y)
    """
    X, y = sample_processed_data
    
    model = RandomForestClassifier(
        n_estimators=10,  # Small for fast testing
        random_state=42,
        max_depth=5
    )
    
    model.fit(X, y)
    return model


@pytest.fixture
def sample_model_file(trained_model, temp_workspace):
    """
    Create a sample model file for testing.
    
    @returns {Path} Path to saved model file
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_model_loading(sample_model_file):
        model = joblib.load(sample_model_file)
        assert hasattr(model, 'predict')
    """
    model_path = temp_workspace / 'models' / 'test_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(trained_model, model_path)
    return model_path


@pytest.fixture
def sample_data_files(sample_raw_data, temp_workspace):
    """
    Create sample data files for testing.
    
    @returns {Dict[str, Path]} Dictionary of data file paths
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_data_files(sample_data_files):
        train_file = sample_data_files['train']
        assert train_file.exists()
    """
    train_df, test_df = sample_raw_data
    
    # Save files
    train_path = temp_workspace / 'data' / 'raw' / 'train.csv'
    test_path = temp_workspace / 'data' / 'raw' / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return {
        'train': train_path,
        'test': test_path,
        'train_df': train_df,
        'test_df': test_df
    }


@pytest.fixture
def sample_config_file(test_config, temp_workspace):
    """
    Create a sample configuration file for testing.
    
    @returns {Path} Path to configuration file
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_config_loading(sample_config_file):
        config = load_config(sample_config_file)
        assert config['project']['name'] == 'healthcare-los-test'
    """
    import yaml
    
    config_path = temp_workspace / 'config.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    return config_path


@pytest.fixture
def mock_predictions():
    """
    Generate mock prediction data for testing.
    
    @returns {Dict[str, Any]} Mock prediction results
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_prediction_format(mock_predictions):
        assert 'predictions' in mock_predictions
        assert len(mock_predictions['predictions']) > 0
    """
    np.random.seed(42)
    n_samples = 100
    
    return {
        'predictions': np.random.randint(0, 4, n_samples).tolist(),
        'probabilities': np.random.random((n_samples, 4)).tolist(),
        'num_predictions': n_samples,
        'prediction_shape': (n_samples,)
    }


@pytest.fixture
def mock_evaluation_results():
    """
    Generate mock evaluation results for testing.
    
    @returns {Dict[str, Any]} Mock evaluation results
    @version 1.0.0
    @public
    
    @example
    # Use in test
    def test_evaluation_format(mock_evaluation_results):
        assert 'accuracy' in mock_evaluation_results
        assert 0 <= mock_evaluation_results['accuracy'] <= 1
    """
    return {
        'accuracy': 0.85,
        'f1_weighted': 0.83,
        'f1_macro': 0.82,
        'f1_micro': 0.85,
        'precision_weighted': 0.84,
        'recall_weighted': 0.83,
        'roc_auc': 0.88,
        'classification_report': {
            '0': {'precision': 0.85, 'recall': 0.80, 'f1-score': 0.82},
            '1': {'precision': 0.82, 'recall': 0.85, 'f1-score': 0.83},
            '2': {'precision': 0.86, 'recall': 0.84, 'f1-score': 0.85},
            '3': {'precision': 0.83, 'recall': 0.82, 'f1-score': 0.82}
        },
        'confusion_matrix': [[80, 5, 3, 2], [6, 85, 4, 5], [2, 4, 84, 10], [3, 6, 8, 83]]
    }


# Pytest marks for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """
    Configure pytest with custom markers.
    
    @param config - Pytest configuration object
    @version 1.0.0
    @public
    """
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on file location.
    
    @param config - Pytest configuration object
    @param items - List of test items
    @version 1.0.0
    @public
    """
    for item in items:
        # Add unit marker to tests in test_unit/ directory
        if "test_unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in test_integration/ directory
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that take longer
        if any(keyword in item.name.lower() for keyword in ['train', 'pipeline', 'end_to_end']):
            item.add_marker(pytest.mark.slow) 