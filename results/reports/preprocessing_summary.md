# Healthcare Length of Stay Preprocessing Implementation

## Overview
This document outlines the preprocessing implementation for the healthcare Length of Stay (LOS) prediction project. The preprocessing handles missing values, outliers, categorical encoding, and feature scaling in a coherent workflow using a single data processing function.

## Data Splitting Approach
- Training data (80%) and validation data (20%) split using stratified sampling based on the target variable 'Stay'
- This ensures all classes are represented proportionally in both datasets
- Stratification is particularly important due to the imbalanced nature of the target variable

## Feature Categories
The features were organized into distinct categories for appropriate processing:

1. **Numerical Features**
   - Features: Available Extra Rooms in Hospital, Visitors with Patient, Admission_Deposit, etc.
   - Treatment: Imputation, outlier capping, standardization

2. **High Cardinality Categorical Features**
   - Features: Hospital_code, City_Code_Hospital, City_Code_Patient
   - Treatment: Imputation, one-hot encoding

3. **Standard Categorical Features**
   - Features: Hospital_type_code, Department, Ward_Type, etc.
   - Treatment: Imputation, one-hot encoding

4. **Age Feature (Special Handling)**
   - Treated separately to convert range categories to numeric midpoints
   - Example: '21-30' → 25.5

## Preprocessing Steps

### Missing Data Handling
- **City_Code_Patient**: Missing values (1.42%) imputed with mode from training data
- **Bed Grade**: Missing values (0.035%) imputed with mode from training data

### Age Feature Processing
- Age ranges converted to numeric midpoint values
- Example: '21-30' is converted to 25.5
- Provides a continuous representation of the originally categorical age feature

### Outlier Treatment
- Implemented capping at 1st and 99th percentiles to handle extreme values
- Applied only to numerical features
- Caps calculated on training data and stored for consistent application to validation/test data

### Categorical Encoding
- One-hot encoding applied to all categorical features
- Implementation ensures proper handling of new categories that might appear in validation/test data
- High cardinality features like Hospital_code encoded with same approach

### Feature Scaling
- Standardization (z-score normalization) applied to all numerical features
- Each feature transformed to have mean=0 and standard deviation=1
- Scaling parameters (mean, std) calculated on training data and stored for validation/test data

## Implementation Approach
- Custom `preprocess_data()` function handles the entire preprocessing workflow
- Parameters learned during training (modes, scaling params) stored in global variables
- Implementation ensures that validation and test data are transformed using parameters from training data only

## Output Artifacts
1. **Preprocessing Parameters**: Serialized parameters (preprocessing_params.joblib)
2. **Processed Data**:
   - X_train_processed.csv - Preprocessed training features
   - X_val_processed.csv - Preprocessed validation features
   - X_test_processed.csv - Preprocessed test features
   - y_train.npy - Training target values
   - y_val.npy - Validation target values
3. **Feature Names**: List of feature names after preprocessing (feature_names.txt)

## Preprocessing Results
- Input dimensions: 17 features
- Output dimensions: 41 features (after one-hot encoding)
- All missing values handled
- All categorical features encoded
- All numerical features normalized
- ID columns removed from processed data

## Next Steps
- Feature engineering to create additional features
- Target encoding for high cardinality features (potential enhancement)
- Model development with handling for class imbalance
- Hyperparameter tuning 