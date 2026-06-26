# Healthcare Length of Stay Prediction - API Documentation

This document provides comprehensive API documentation for all modules, classes, and functions in the Healthcare Length of Stay Prediction project.

## Table of Contents

- [Data Module](#data-module)
- [Models Module](#models-module)
- [Evaluation Module](#evaluation-module)
- [Visualization Module](#visualization-module)
- [Utils Module](#utils-module)
- [Scripts](#scripts)

---

## Data Module

### `src.data.make_dataset`

**Description**: Core data loading and dataset creation utilities.

#### Functions

##### `load_raw_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]`

Load raw training and test datasets from files.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing data paths

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Training and test dataframes

**Raises:**
- `FileNotFoundError`: When data files are not found
- `ValueError`: When data format is invalid

**Example:**
```python
from src.data.make_dataset import load_raw_data
from src.utils.config import load_config

config = load_config()
train_df, test_df = load_raw_data(config)
print(f"Training data shape: {train_df.shape}")
```

##### `load_processed_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]`

Load preprocessed features and target data.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Processed features (X) and target (y)

**Example:**
```python
X, y = load_processed_data(config)
print(f"Features shape: {X.shape}, Target shape: {y.shape}")
```

##### `validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]`

Perform comprehensive data quality validation.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate

**Returns:**
- `Dict[str, Any]`: Quality report with missing values, duplicates, and data types

**Example:**
```python
quality_report = validate_data_quality(train_df)
print(f"Missing values: {quality_report['missing_values']}")
print(f"Duplicates: {quality_report['duplicates']}")
```

##### `get_target_distribution(target: pd.Series) -> Dict[str, Any]`

Analyze target variable distribution.

**Parameters:**
- `target` (pd.Series): Target variable series

**Returns:**
- `Dict[str, Any]`: Distribution statistics and value counts

**Example:**
```python
distribution = get_target_distribution(train_df['lengthofstay'])
print(f"Class distribution: {distribution['value_counts']}")
```

##### `create_sample_data(df: pd.DataFrame, sample_rate: float = 0.1, stratify_column: str = None, random_state: int = 42) -> pd.DataFrame`

Create sample datasets for testing or development.

**Parameters:**
- `df` (pd.DataFrame): Source dataframe
- `sample_rate` (float): Proportion of data to sample (0.0-1.0)
- `stratify_column` (str, optional): Column for stratified sampling
- `random_state` (int): Random seed for reproducibility

**Returns:**
- `pd.DataFrame`: Sampled dataframe

---

### `src.data.preprocessing`

**Description**: Data preprocessing and feature engineering utilities.

#### Functions

##### `preprocess_features(X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]`

Apply comprehensive feature preprocessing pipeline.

**Parameters:**
- `X` (pd.DataFrame): Raw features
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `Tuple[pd.DataFrame, List[str]]`: Processed features and list of applied preprocessing steps

**Example:**
```python
from src.data.preprocessing import preprocess_features

X_processed, steps = preprocess_features(X_raw, config)
print(f"Preprocessing steps applied: {steps}")
```

##### `split_data(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]`

Split data into training, validation, and test sets.

**Parameters:**
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target variable
- `config` (Dict[str, Any]): Configuration with split parameters

**Returns:**
- `Dict[str, Any]`: Dictionary containing train/test/validation splits

**Example:**
```python
from src.data.preprocessing import split_data

splits = split_data(X, y, config)
X_train, X_test = splits['X_train'], splits['X_test']
y_train, y_test = splits['y_train'], splits['y_test']
```

##### `scale_features(X: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Any]`

Scale numerical features using specified method.

**Parameters:**
- `X` (pd.DataFrame): Features to scale
- `method` (str): Scaling method ('standard', 'minmax', 'robust')

**Returns:**
- `Tuple[pd.DataFrame, Any]`: Scaled features and fitted scaler

##### `encode_categorical_features(X: pd.DataFrame, categorical_columns: List[str], method: str = 'onehot') -> Tuple[pd.DataFrame, Dict]`

Encode categorical features using specified method.

**Parameters:**
- `X` (pd.DataFrame): Features containing categorical columns
- `categorical_columns` (List[str]): List of categorical column names
- `method` (str): Encoding method ('onehot', 'label', 'target')

**Returns:**
- `Tuple[pd.DataFrame, Dict]`: Encoded features and fitted encoders

##### `handle_missing_values(X: pd.DataFrame, strategy: str = 'mean') -> Tuple[pd.DataFrame, Dict]`

Handle missing values using specified strategy.

**Parameters:**
- `X` (pd.DataFrame): Features with missing values
- `strategy` (str): Imputation strategy ('mean', 'median', 'mode', 'constant')

**Returns:**
- `Tuple[pd.DataFrame, Dict]`: Imputed features and fitted imputers

---

## Models Module

### `src.models.base_model`

**Description**: Base model class providing common interface for all models.

#### Classes

##### `BaseModel`

Abstract base class for all machine learning models.

**Methods:**

###### `train(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]`

Train the model with given data.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training targets
- `X_val` (pd.DataFrame, optional): Validation features
- `y_val` (pd.Series, optional): Validation targets

**Returns:**
- `Dict[str, Any]`: Training results and metrics

###### `predict(X: pd.DataFrame) -> np.ndarray`

Make predictions on given data.

**Parameters:**
- `X` (pd.DataFrame): Features for prediction

**Returns:**
- `np.ndarray`: Predictions

###### `predict_proba(X: pd.DataFrame) -> np.ndarray`

Get prediction probabilities (if supported).

**Parameters:**
- `X` (pd.DataFrame): Features for prediction

**Returns:**
- `np.ndarray`: Prediction probabilities

###### `save_model(path: str) -> None`

Save the trained model to file.

**Parameters:**
- `path` (str): File path to save model

###### `load_model(path: str) -> None`

Load a trained model from file.

**Parameters:**
- `path` (str): File path to load model from

---

### `src.models.train_model`

**Description**: Model training orchestration and management.

#### Classes

##### `ModelTrainer`

Coordinates training of multiple models.

**Methods:**

###### `__init__(config: Dict[str, Any])`

Initialize the model trainer.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

###### `train_all_models(data_splits: Dict[str, Any]) -> Dict[str, Any]`

Train all configured models.

**Parameters:**
- `data_splits` (Dict[str, Any]): Data splits from preprocessing

**Returns:**
- `Dict[str, Any]`: Training results for all models

###### `train_selected_models(data_splits: Dict[str, Any], model_names: List[str]) -> Dict[str, Any]`

Train specified models only.

**Parameters:**
- `data_splits` (Dict[str, Any]): Data splits from preprocessing
- `model_names` (List[str]): List of model names to train

**Returns:**
- `Dict[str, Any]`: Training results for selected models

---

## Evaluation Module

### `src.evaluation.metrics`

**Description**: Model evaluation metrics and comparison utilities.

#### Functions

##### `calculate_comprehensive_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]`

Calculate comprehensive evaluation metrics for a model.

**Parameters:**
- `model` (Any): Trained model object
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test targets
- `X_val` (pd.DataFrame, optional): Validation features
- `y_val` (pd.Series, optional): Validation targets

**Returns:**
- `Dict[str, Any]`: Comprehensive metrics including accuracy, F1, precision, recall, ROC-AUC

**Example:**
```python
from src.evaluation.metrics import calculate_comprehensive_metrics

metrics = calculate_comprehensive_metrics(model, X_test, y_test)
print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
print(f"Test F1 Score: {metrics['test_f1_weighted']:.3f}")
```

##### `compare_models(evaluation_results: Dict[str, Any]) -> Dict[str, Any]`

Compare multiple model evaluation results.

**Parameters:**
- `evaluation_results` (Dict[str, Any]): Results from multiple model evaluations

**Returns:**
- `Dict[str, Any]`: Comparison results with rankings and best model

**Example:**
```python
comparison = compare_models(evaluation_results)
print(f"Best model: {comparison['best_model']}")
print(f"Model rankings: {comparison['rankings']}")
```

---

## Visualization Module

### `src.visualization.model_plots`

**Description**: Visualization utilities for model analysis and reporting.

#### Functions

##### `plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None, save_path: str = None) -> matplotlib.figure.Figure`

Create confusion matrix visualization.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `class_names` (List[str], optional): Class names for labels
- `save_path` (str, optional): Path to save plot

**Returns:**
- `matplotlib.figure.Figure`: Matplotlib figure object

##### `plot_roc_curves(models_results: Dict[str, Any], save_path: str = None) -> matplotlib.figure.Figure`

Create ROC curve comparison plot.

**Parameters:**
- `models_results` (Dict[str, Any]): Results from multiple models
- `save_path` (str, optional): Path to save plot

**Returns:**
- `matplotlib.figure.Figure`: Matplotlib figure object

##### `generate_evaluation_plots(evaluation_results: Dict[str, Any], data_splits: Dict[str, Any], output_dir: Path) -> Dict[str, Any]`

Generate comprehensive evaluation plots for all models.

**Parameters:**
- `evaluation_results` (Dict[str, Any]): Model evaluation results
- `data_splits` (Dict[str, Any]): Data splits for plotting
- `output_dir` (Path): Directory to save plots

**Returns:**
- `Dict[str, Any]`: Information about generated plots

---

## Utils Module

### `src.utils.config`

**Description**: Configuration management utilities.

#### Functions

##### `load_config(config_path: str = None) -> Dict[str, Any]`

Load configuration from YAML file.

**Parameters:**
- `config_path` (str, optional): Path to configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

**Example:**
```python
from src.utils.config import load_config

config = load_config('config/config.yaml')
print(f"Project name: {config['project']['name']}")
```

##### `setup_logging(config: Dict[str, Any] = None) -> None`

Set up logging configuration.

**Parameters:**
- `config` (Dict[str, Any], optional): Configuration dictionary

**Example:**
```python
from src.utils.config import setup_logging

setup_logging(config)
logger = logging.getLogger(__name__)
logger.info("Logging configured successfully")
```

##### `get_data_path(config: Dict[str, Any]) -> Path`

Get data directory path from configuration.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `Path`: Data directory path

##### `get_model_path(config: Dict[str, Any]) -> Path`

Get models directory path from configuration.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `Path`: Models directory path

##### `get_results_path(config: Dict[str, Any]) -> Path`

Get results directory path from configuration.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `Path`: Results directory path

---

## Scripts

### `scripts/train_model.py`

**Description**: Command-line script for training models.

**Usage:**
```bash
python scripts/train_model.py [OPTIONS]
```

**Options:**
- `--config`: Path to configuration file
- `--model`: Specific model to train
- `--all`: Train all models
- `--hyperparameter-tuning`: Enable hyperparameter tuning
- `--cross-validation`: Enable cross-validation
- `--verbose`: Enable verbose logging

**Examples:**
```bash
# Train a specific model
python scripts/train_model.py --model xgboost --config config/config.yaml

# Train all models with hyperparameter tuning
python scripts/train_model.py --all --hyperparameter-tuning

# Train with cross-validation
python scripts/train_model.py --model lightgbm --cross-validation --verbose
```

---

### `scripts/evaluate_model.py`

**Description**: Command-line script for model evaluation.

**Usage:**
```bash
python scripts/evaluate_model.py [OPTIONS]
```

**Options:**
- `--model-dir`: Path to model directory
- `--model-file`: Path to specific model file
- `--compare-all`: Compare all models
- `--output`: Output file for results
- `--verbose`: Enable verbose logging

**Examples:**
```bash
# Evaluate a specific model
python scripts/evaluate_model.py --model-file models/xgboost/model.pkl

# Compare all models
python scripts/evaluate_model.py --compare-all --output results/comparison.json

# Evaluate models in directory
python scripts/evaluate_model.py --model-dir models/ensemble --verbose
```

---

### `scripts/predict.py`

**Description**: Command-line script for making predictions.

**Usage:**
```bash
python scripts/predict.py [OPTIONS]
```

**Options:**
- `--model`: Path to model file
- `--model-dir`: Path to model directory (uses latest)
- `--input`: Path to input data file
- `--json`: JSON string for single prediction
- `--output`: Output file path
- `--format`: Output format (csv, json, excel, parquet)
- `--probabilities`: Include prediction probabilities
- `--no-preprocessing`: Skip preprocessing

**Examples:**
```bash
# Batch prediction
python scripts/predict.py --model models/xgboost/model.pkl --input data/new_patients.csv

# Single prediction with probabilities
python scripts/predict.py --model models/xgboost/model.pkl --json '{"age": 65, "severity": "high"}' --probabilities

# Output as JSON with probabilities
python scripts/predict.py --model-dir models/ensemble --input data/test.csv --format json --probabilities --output predictions.json
```

---

### `scripts/run_pipeline.py`

**Description**: Command-line script for running the complete ML pipeline.

**Usage:**
```bash
python scripts/run_pipeline.py [OPTIONS]
```

**Options:**
- `--config`: Path to configuration file
- `--models`: Comma-separated list of models to train
- `--skip-preprocessing`: Skip preprocessing step
- `--force-reload`: Force reload of raw data
- `--skip-evaluation`: Skip evaluation step
- `--skip-reporting`: Skip reporting step
- `--verbose`: Enable verbose logging

**Examples:**
```bash
# Run full pipeline
python scripts/run_pipeline.py --config config/config.yaml

# Run with specific models
python scripts/run_pipeline.py --models xgboost,lightgbm,ensemble

# Skip preprocessing and force data reload
python scripts/run_pipeline.py --skip-preprocessing --force-reload --verbose

# Run training only (skip evaluation and reporting)
python scripts/run_pipeline.py --skip-evaluation --skip-reporting
```

---

## Error Handling

All functions include comprehensive error handling with informative error messages. Common exceptions include:

- `FileNotFoundError`: When required files are not found
- `ValueError`: When invalid parameters or data are provided
- `ConfigurationError`: When configuration is invalid or incomplete
- `ModelError`: When model training or prediction fails

## Performance Considerations

- Use `create_sample_data()` for development and testing with large datasets
- Enable verbose logging only when needed as it can impact performance
- Consider using `--skip-preprocessing` when data is already processed
- Use multiprocessing capabilities in scripts for faster execution on multi-core systems

## Version Compatibility

- Python: 3.8+
- pandas: 1.5+
- scikit-learn: 1.0+
- All other dependencies as specified in `requirements.txt` 