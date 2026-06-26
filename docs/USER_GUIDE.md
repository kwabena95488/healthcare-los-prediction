# Healthcare Length of Stay Prediction - User Guide

This comprehensive guide will help you get started with the Healthcare Length of Stay Prediction system, from basic setup to advanced workflows.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation Guide](#installation-guide)
- [Basic Usage](#basic-usage)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Making Predictions](#making-predictions)
- [Full Pipeline Workflow](#full-pipeline-workflow)
- [Configuration Guide](#configuration-guide)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Quick Start

### 5-Minute Setup

1. **Clone and install**
   ```bash
   git clone <repository-url>
   cd Healthcare-Analytics-main
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Verify installation**
   ```bash
   python validate_setup.py
   ```

3. **Run a quick prediction**
   ```bash
   python scripts/predict.py --model-dir models --json '{"age": 65, "severity": "high"}'
   ```

### What You Can Do

- 🔄 **Train Models**: Train multiple ML models (XGBoost, LightGBM, etc.)
- 📊 **Evaluate Performance**: Compare models with comprehensive metrics
- 🎯 **Make Predictions**: Batch or single predictions with confidence scores
- 🔬 **Run Full Pipeline**: End-to-end ML workflow automation
- 📈 **Visualize Results**: Generate plots and analysis reports

---

## Installation Guide

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for dependencies and models

### Installation Options

#### Option 1: Standard Installation

```bash
# Clone repository
git clone <repository-url>
cd Healthcare-Analytics-main

# Create virtual environment (recommended)
python -m venv healthcare_env
source healthcare_env/bin/activate  # On Windows: healthcare_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

#### Option 2: Development Installation

```bash
# For contributors and developers
git clone <repository-url>
cd Healthcare-Analytics-main

# Install with development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Verify development setup
python -m pytest tests/ -v
```

#### Option 3: Makefile Installation

```bash
# If you have Make installed
make install
make test
```

### Verification

Run the validation script to ensure everything is working:

```bash
python validate_setup.py
```

You should see all checks passing (✅).

---

## Basic Usage

### Understanding the CLI Scripts

The system provides four main command-line tools:

1. **`scripts/train_model.py`** - Train machine learning models
2. **`scripts/evaluate_model.py`** - Evaluate and compare models  
3. **`scripts/predict.py`** - Make predictions on new data
4. **`scripts/run_pipeline.py`** - Run the complete ML pipeline

### Getting Help

Each script provides comprehensive help:

```bash
python scripts/train_model.py --help
python scripts/evaluate_model.py --help
python scripts/predict.py --help
python scripts/run_pipeline.py --help
```

### Basic Command Structure

All scripts follow a consistent pattern:

```bash
python scripts/<script_name>.py [OPTIONS]
```

Common options:
- `--config`: Specify configuration file
- `--verbose`: Enable detailed logging
- `--help`: Show help message

---

## Data Preparation

### Data Format Requirements

The system expects healthcare data with the following characteristics:

- **Format**: CSV files with headers
- **Target Variable**: `lengthofstay` column (can be configured)
- **Features**: Patient demographics, medical history, procedures, etc.

### Example Data Structure

```csv
age,gender,admission_type,severity_illness,risk_mortality,lengthofstay
65,1,1,3,2,2
45,0,2,1,1,0
72,1,1,4,3,3
...
```

### Data Placement

Organize your data in the project structure:

```
data/
├── raw/                    # Original datasets
│   ├── train.csv          # Training data
│   ├── test.csv           # Test data (optional)
│   └── column-descriptions.md
├── processed/             # Cleaned data (auto-generated)
└── interim/              # Intermediate files
```

### Loading Your Data

1. **Place raw data files in `data/raw/`**
2. **Update configuration** (optional - see Configuration Guide)
3. **The system will handle preprocessing automatically**

---

## Model Training

### Train a Single Model

```bash
# Train XGBoost model
python scripts/train_model.py --model xgboost --config config/config.yaml

# Train with verbose logging
python scripts/train_model.py --model lightgbm --verbose
```

### Train All Models

```bash
# Train all configured models
python scripts/train_model.py --all

# Train with hyperparameter tuning
python scripts/train_model.py --all --hyperparameter-tuning
```

### Train Specific Models

```bash
# Train multiple specific models
python scripts/train_model.py --models xgboost,lightgbm,catboost
```

### Training with Cross-Validation

```bash
# Enable cross-validation for robust evaluation
python scripts/train_model.py --model xgboost --cross-validation --folds 5
```

### Monitor Training Progress

Training progress is logged in real-time:

```
==========================================
TRAINING: XGBoost Model
==========================================
Loading data...
Data loaded: (800, 20) training samples
Preprocessing features...
Training XGBoost model...
Training completed in 45.2 seconds
Model saved to: models/xgboost/model_20241219_143022.pkl
Validation score: 0.847
```

### Training Outputs

After training, you'll find:

```
models/
├── xgboost/
│   ├── model_20241219_143022.pkl    # Trained model
│   ├── hyperparameters.json        # Used parameters
│   └── training_log.txt             # Training details
└── experiments/
    └── mlruns/                      # MLflow tracking
```

---

## Model Evaluation

### Evaluate a Specific Model

```bash
# Evaluate single model
python scripts/evaluate_model.py --model-file models/xgboost/model.pkl

# Evaluate latest model in directory
python scripts/evaluate_model.py --model-dir models/xgboost
```

### Compare All Models

```bash
# Compare all trained models
python scripts/evaluate_model.py --compare-all

# Save comparison results
python scripts/evaluate_model.py --compare-all --output results/model_comparison.json
```

### Evaluation Metrics

The system calculates comprehensive metrics:

- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Weighted, macro, and micro F1 scores
- **Precision & Recall**: Weighted averages
- **ROC AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed prediction breakdown
- **Classification Report**: Per-class performance

### Sample Evaluation Output

```
==================================================
MODEL EVALUATION SUMMARY
==================================================
Total models evaluated: 4
Best model: xgboost
Best F1 Score: 0.847

Model Rankings:
1. xgboost - F1: 0.847, Accuracy: 0.852
2. lightgbm - F1: 0.841, Accuracy: 0.845
3. catboost - F1: 0.835, Accuracy: 0.840
4. random_forest - F1: 0.798, Accuracy: 0.805

Results saved to: results/metrics/evaluation_results.json
```

---

## Making Predictions

### Batch Predictions

Predict on multiple samples from a file:

```bash
# Basic batch prediction
python scripts/predict.py --model models/xgboost/model.pkl --input data/new_patients.csv

# Include prediction probabilities
python scripts/predict.py --model models/xgboost/model.pkl --input data/test.csv --probabilities

# Custom output format and location
python scripts/predict.py --model models/best_model.pkl --input data/patients.csv --output predictions.json --format json
```

### Single Predictions

Make a prediction for one patient:

```bash
# Single prediction with JSON input
python scripts/predict.py --model models/xgboost/model.pkl --json '{"age": 65, "gender": 1, "severity_illness": 3}'

# With probabilities
python scripts/predict.py --model models/xgboost/model.pkl --json '{"age": 45, "admission_type": 2}' --probabilities
```

### Supported Input Formats

- **CSV**: `.csv` files
- **JSON**: `.json` files  
- **Excel**: `.xlsx`, `.xls` files
- **Parquet**: `.parquet` files

### Supported Output Formats

- **CSV**: Default format with predictions column
- **JSON**: Structured JSON with metadata
- **Excel**: Excel file with predictions
- **Parquet**: Compressed columnar format

### Sample Prediction Output

```bash
$ python scripts/predict.py --model models/xgboost/model.pkl --json '{"age": 65, "severity": "high"}' --probabilities

==================================================
SINGLE PREDICTION RESULT
==================================================
Prediction: 2
Probabilities: [0.05, 0.15, 0.65, 0.15]
```

For batch predictions, output files include original data plus predictions:

```csv
age,gender,admission_type,prediction,probability_class_0,probability_class_1,probability_class_2,probability_class_3
65,1,1,2,0.05,0.15,0.65,0.15
45,0,2,1,0.1,0.7,0.15,0.05
...
```

---

## Full Pipeline Workflow

### Run Complete Pipeline

The pipeline script orchestrates the entire ML workflow:

```bash
# Run full pipeline with default settings
python scripts/run_pipeline.py

# Run with specific models
python scripts/run_pipeline.py --models xgboost,lightgbm,ensemble

# Skip preprocessing if data is already clean
python scripts/run_pipeline.py --skip-preprocessing
```

### Pipeline Stages

1. **Data Loading**: Load and validate raw data
2. **Preprocessing**: Clean, transform, and split data
3. **Training**: Train selected models
4. **Evaluation**: Evaluate and compare models
5. **Reporting**: Generate plots and reports

### Pipeline Configuration

```bash
# Use custom configuration
python scripts/run_pipeline.py --config config/my_config.yaml

# Force reload of data
python scripts/run_pipeline.py --force-reload

# Skip specific stages
python scripts/run_pipeline.py --skip-evaluation --skip-reporting
```

### Pipeline Output

The pipeline generates comprehensive outputs:

```
results/
├── figures/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── feature_importance/
├── metrics/
│   └── evaluation_results.json
└── reports/
    ├── pipeline_20241219_143022_report.json
    └── pipeline_20241219_143022_summary.txt
```

### Sample Pipeline Summary

```
======================================================
PIPELINE EXECUTION SUMMARY
======================================================
Pipeline ID: pipeline_20241219_143022
Total execution time: 245.67 seconds
Status: SUCCESS

PIPELINE STEPS SUMMARY
=====================

DATA_LOADING: SUCCESS (12.45s)
  - Data shape: (1000, 21)
  - Data source: raw

PREPROCESSING: SUCCESS (34.12s)
  - Data splits - Train: 640, Val: 160, Test: 200

TRAINING: SUCCESS (156.23s)
  - Models trained: 4
  - Models failed: 0

EVALUATION: SUCCESS (28.45s)
  - Models evaluated: 4
  - Best model: xgboost
  - Best score: 0.847

REPORTING: SUCCESS (14.42s)
  - Plots generated: 12
```

---

## Configuration Guide

### Configuration Files

The system uses YAML configuration files:

- **`config/config.yaml`**: Main project configuration
- **`config/model_config.yaml`**: Model hyperparameters
- **`config/logging.yaml`**: Logging settings

### Main Configuration (`config/config.yaml`)

```yaml
project:
  name: "healthcare-length-of-stay"
  version: "1.0.0"

data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  target_column: "lengthofstay"
  test_size: 0.2
  val_size: 0.2
  random_state: 42

models:
  output_path: "models"
  random_state: 42
  enabled:
    - "xgboost"
    - "lightgbm"
    - "catboost"
    - "random_forest"

results:
  output_path: "results"
```

### Model Configuration (`config/model_config.yaml`)

```yaml
xgboost:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8

lightgbm:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  feature_fraction: 0.8
```

### Custom Configuration

```bash
# Use custom configuration file
python scripts/train_model.py --config my_custom_config.yaml

# Override specific settings with environment variables
export DATA_PATH="/path/to/my/data"
python scripts/run_pipeline.py
```

---

## Common Use Cases

### Use Case 1: Quick Model Comparison

**Goal**: Compare different algorithms on your data

```bash
# 1. Train multiple models
python scripts/train_model.py --models xgboost,lightgbm,catboost,random_forest

# 2. Compare performance
python scripts/evaluate_model.py --compare-all --output model_comparison.json

# 3. View results
cat model_comparison.json
```

### Use Case 2: Production Model Training

**Goal**: Train a robust model for production use

```bash
# 1. Train with cross-validation and hyperparameter tuning
python scripts/train_model.py --model xgboost --cross-validation --hyperparameter-tuning

# 2. Evaluate thoroughly
python scripts/evaluate_model.py --model-dir models/xgboost --verbose

# 3. Test predictions
python scripts/predict.py --model models/xgboost/best_model.pkl --input data/validation.csv
```

### Use Case 3: Automated Pipeline

**Goal**: Set up automated training and evaluation

```bash
# Run complete pipeline with comprehensive reporting
python scripts/run_pipeline.py --models xgboost,lightgbm --verbose

# Results automatically saved to results/ directory
ls results/reports/
```

### Use Case 4: Batch Prediction Service

**Goal**: Process large batches of patient data

```bash
# Process multiple files
for file in data/batches/*.csv; do
    python scripts/predict.py --model models/production/best_model.pkl --input "$file" --output "predictions/$(basename "$file")"
done
```

### Use Case 5: Model Monitoring

**Goal**: Monitor model performance over time

```bash
# Weekly evaluation
python scripts/evaluate_model.py --model-file models/production/current_model.pkl --output "monitoring/eval_$(date +%Y%m%d).json"

# Compare with baseline
python scripts/evaluate_model.py --compare-all --output "monitoring/comparison_$(date +%Y%m%d).json"
```

---

## Troubleshooting

### Common Issues

#### 1. Installation Problems

**Problem**: `pip install` fails with dependency conflicts

**Solution**:
```bash
# Use virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Problem**: Missing system dependencies

**Solution**:
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# On macOS
xcode-select --install
brew install python
```

#### 2. Data Loading Issues

**Problem**: `FileNotFoundError` when loading data

**Solution**:
```bash
# Check data file paths
ls data/raw/
python validate_setup.py

# Update configuration if needed
vim config/config.yaml
```

**Problem**: Data format errors

**Solution**:
```python
# Check your CSV format
import pandas as pd
df = pd.read_csv('data/raw/train.csv')
print(df.head())
print(df.info())
```

#### 3. Training Failures

**Problem**: Model training fails with memory errors

**Solution**:
```bash
# Use data sampling for large datasets
python scripts/train_model.py --sample-rate 0.5 --model xgboost

# Or increase system memory / use cloud instance
```

**Problem**: Poor model performance

**Solution**:
```bash
# Enable hyperparameter tuning
python scripts/train_model.py --model xgboost --hyperparameter-tuning

# Try ensemble methods
python scripts/train_model.py --model ensemble
```

#### 4. Prediction Issues

**Problem**: Prediction format errors

**Solution**:
```bash
# Check input data format
head data/new_patients.csv

# Validate column names match training data
python -c "import pandas as pd; print(pd.read_csv('data/new_patients.csv').columns.tolist())"
```

#### 5. Configuration Problems

**Problem**: YAML configuration errors

**Solution**:
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Use default configuration
cp config/config.yaml.example config/config.yaml
```

### Getting Detailed Logs

Enable verbose logging for debugging:

```bash
# Enable verbose mode
python scripts/run_pipeline.py --verbose

# Check log files
tail -f logs/healthcare_los.log
```

### Performance Optimization

#### Speed Up Training

```bash
# Use subset of data for testing
python scripts/train_model.py --sample-rate 0.1 --model xgboost

# Reduce cross-validation folds
python scripts/train_model.py --cross-validation --folds 3
```

#### Reduce Memory Usage

```bash
# Process data in chunks
python scripts/predict.py --model models/xgboost/model.pkl --input large_dataset.csv --batch-size 1000
```

---

## FAQ

### General Questions

**Q: What types of healthcare data does this system support?**

A: The system is designed for healthcare length of stay prediction but can work with any tabular healthcare data including patient demographics, medical history, procedures, diagnoses, and lab values.

**Q: Can I use this with my own dataset?**

A: Yes! Place your CSV files in `data/raw/` and update the target column name in `config/config.yaml` if needed.

**Q: How accurate are the predictions?**

A: Accuracy depends on your data quality and size. The system typically achieves 80-90% accuracy on well-prepared healthcare datasets.

### Technical Questions

**Q: Which machine learning algorithms are supported?**

A: Currently supported:
- XGBoost
- LightGBM  
- CatBoost
- Random Forest
- Neural Networks
- Ensemble methods

**Q: Can I add custom models?**

A: Yes! Extend the `BaseModel` class in `src/models/base_model.py` and add your implementation.

**Q: How do I handle missing data?**

A: The preprocessing pipeline automatically handles missing values using configurable imputation strategies (mean, median, mode, or constant values).

**Q: Can I deploy this in production?**

A: Yes! The system is designed for production use with:
- CLI scripts for automation
- Configuration management
- Model versioning
- Comprehensive logging
- Testing framework

### Data Questions

**Q: What's the minimum dataset size needed?**

A: Recommended minimum:
- 1000+ samples for reliable results
- 10+ samples per class for classification
- More data generally improves performance

**Q: How do I handle categorical variables?**

A: The system automatically detects and encodes categorical variables using one-hot encoding, label encoding, or target encoding based on configuration.

**Q: Can I use external data sources?**

A: Yes! Extend the data loading functions in `src/data/make_dataset.py` to support databases, APIs, or other data sources.

### Usage Questions

**Q: How do I retrain models with new data?**

A: Simply add new data to `data/raw/` and run:
```bash
python scripts/train_model.py --all --force-reload
```

**Q: How do I schedule regular retraining?**

A: Use cron jobs or task schedulers:
```bash
# Add to crontab for weekly retraining
0 2 * * 0 cd /path/to/project && python scripts/run_pipeline.py
```

**Q: Can I compare models from different time periods?**

A: Yes! The system timestamps all models and results. Use the evaluation script to compare:
```bash
python scripts/evaluate_model.py --model-dir models/archive --compare-all
```

### Support Questions

**Q: Where can I get help?**

A: Check the documentation in order:
1. This user guide
2. API documentation (`docs/API.md`)  
3. Contributing guide (`docs/CONTRIBUTING.md`)
4. GitHub issues for bug reports
5. Code examples in `scripts/` directory

**Q: How do I report bugs?**

A: Use the GitHub issue template with:
- Clear description of the problem
- Steps to reproduce
- System information
- Error messages/logs

**Q: How do I request new features?**

A: Submit a feature request with:
- Description of the proposed feature
- Use case and motivation
- Suggested implementation approach

---

## Next Steps

### Beginner Path

1. ✅ **Complete installation** and run validation
2. 🎯 **Try basic prediction** with sample data
3. 📊 **Run model comparison** to understand performance
4. 📖 **Explore configuration** options

### Intermediate Path

1. 🔄 **Run full pipeline** with your data
2. ⚙️ **Customize configuration** for your use case
3. 🧪 **Experiment with hyperparameters**
4. 📈 **Generate comprehensive reports**

### Advanced Path

1. 🛠 **Extend the system** with custom models
2. 🚀 **Set up production deployment**
3. 📊 **Implement monitoring** and retraining
4. 🤝 **Contribute** to the project

### Resources for Learning

- **Healthcare AI**: Understanding healthcare data challenges
- **Machine Learning**: scikit-learn, XGBoost documentation
- **MLOps**: Model deployment and monitoring best practices
- **Python**: Advanced Python for data science

---

**Happy modeling! 🏥📊🚀**

*For additional support, please refer to the project documentation or submit an issue on GitHub.* 