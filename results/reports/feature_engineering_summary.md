# Healthcare Length of Stay Feature Engineering Summary

## Overview
This document outlines the feature engineering implementation for the healthcare Length of Stay (LOS) prediction project. The feature engineering process creates additional features to enhance the predictive power of the models.

## Feature Engineering Approaches

### Count/Frequency Features
- **Patient Count**: Number of occurrences of each patient ID in the training data
- **Patient-Region Count**: Number of occurrences of each patient ID within each hospital region
- **Patient-Ward Count**: Number of occurrences of each patient ID within each ward facility type
- **Implementation**: Features calculated on training data only and applied consistently to validation and test sets
- **Missing Value Handling**: Unseen combinations in validation/test set filled with 1

### Interaction Features
- **Room-Visitor Interaction**: Product of 'Available Extra Rooms in Hospital' and 'Visitors with Patient'
- **Deposit-Bed Grade Interaction**: Product of 'Admission_Deposit' and 'Bed Grade'
- **Hospital-City Code Interaction**: Product of 'Hospital_code' and 'City_Code_Hospital'
- **Rationale**: Capturing potential synergistic effects between related features
- **Implementation**: Simple multiplication of numeric features

### Aggregation-based Features
- **Hospital-based Aggregations**: Mean, median, and standard deviation of 'Admission_Deposit' for each hospital
- **City-based Aggregations**: Mean, median, and standard deviation of 'Admission_Deposit' for each city
- **Department-based Aggregations**: Mean, median, and standard deviation of 'Admission_Deposit' for each department
- **Implementation**: Aggregations calculated on training data only and applied consistently to validation and test sets
- **Missing Value Handling**: Unseen combinations in validation/test set filled with global statistics from training data

## Feature Importance Analysis
Feature importance was calculated using mutual information classification to identify the most predictive features:

### Top 5 Most Important Features
1. **Room-Visitor Interaction** (MI Score: 0.221) - Interaction between available rooms and visitors
2. **Visitors with Patient** (MI Score: 0.175) - Original feature showing strong predictive power
3. **Deposit-Bed Grade Interaction** (MI Score: 0.037) - Interaction between admission deposit and bed grade
4. **Admission_Deposit** (MI Score: 0.030) - Original feature with moderate importance
5. **Hospital_code** (MI Score: 0.029) - Hospital identifier showing moderate importance

### Bottom 5 Least Important Features
1. **Department_anesthesia** (MI Score: 0.000) - One-hot encoded feature with minimal predictive power
2. **Department_surgery** (MI Score: 0.000) - One-hot encoded feature with minimal predictive power
3. **Hospital_type_code_e** (MI Score: 0.000) - One-hot encoded feature with minimal predictive power
4. **Hospital_type_code_d** (MI Score: 0.000) - One-hot encoded feature with minimal predictive power
5. **Hospital_type_code_c** (MI Score: 0.000) - One-hot encoded feature with minimal predictive power

## Results and Output
- **Original Features**: 41 features after preprocessing
- **Engineered Features**: 15 additional features created
- **Total Features**: 56 features available for modeling
- **Output Files**:
  - X_train_enriched.csv - Training features with engineered features
  - X_val_enriched.csv - Validation features with engineered features
  - X_test_enriched.csv - Test features with engineered features
  - feature_importance.csv - Feature importance scores
  - feature_names_enriched.txt - List of all feature names

## Key Insights
- **Interaction Features**: The interaction between available rooms and visitors is the most predictive feature, suggesting that hospital capacity and visitor presence together strongly influence length of stay
- **Original Features**: 'Visitors with Patient' remains a strong predictor even after feature engineering
- **Department Features**: Many department one-hot encoded features show minimal predictive power individually
- **Aggregated Statistics**: Hospital-level statistics for admission deposits provide moderate predictive power

## Next Steps
- Model development with the enriched feature set
- Handling class imbalance in the target variable
- Hyperparameter tuning for optimal model performance
- Potential feature selection to remove low-importance features 