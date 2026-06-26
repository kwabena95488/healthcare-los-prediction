# Healthcare Analytics: Length of Stay Prediction - Detailed Implementation Plan

This document outlines a detailed plan for developing and improving the model to predict patient Length of Stay (LOS) in healthcare facilities.

## Phase 1: Project Setup & Advanced Data Understanding

1.  **Environment Configuration:**
    *   **Action:** Create a new Python virtual environment (e.g., `python -m venv .venv` or `conda create -n healthcare_los python=3.9`).
    *   **Action:** Activate the environment (e.g., `source .venv/bin/activate` or `conda activate healthcare_los`).
    *   **Action:** Create a `requirements.txt` file.
    *   **Action:** Install core libraries:
        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
        # Add to requirements.txt
        ```
    *   **Tools:** `venv`/`conda`, `pip`.

2.  **Initial Data Ingestion & Automated Profiling:**
    *   **Action:** Load `train.csv` and `test.csv` into pandas DataFrames.
    *   **Action:** Generate an automated data profiling report for both datasets.
        *   **Tool:** `pandas-profiling` or `sweetviz`.
        *   **Command (example):**
            ```python
            # import pandas_profiling
            # train_df = pd.read_csv('train.csv')
            # profile = pandas_profiling.ProfileReport(train_df, title="Training Data Profile")
            # profile.to_file("train_data_profile.html")
            ```
    *   **Deliverable:** `train_data_profile.html`, `test_data_profile.html`.

3.  **In-Depth Exploratory Data Analysis (EDA):**
    *   **Target Variable ('Stay') Analysis:**
        *   **Action:** Plot the distribution of the 'Stay' column (bar chart of value counts).
        *   **Action:** Calculate and note class frequencies and identify potential imbalance.
    *   **Numerical Feature Analysis:**
        *   **Action:** Generate histograms and box plots for `Available Extra Rooms in Hospital`, `Visitors with Patient`, `Admission_Deposit`.
        *   **Action:** Calculate descriptive statistics (`.describe()`).
        *   **Action:** Visualize relationships with 'Stay' (e.g., box plots of `Admission_Deposit` vs. each 'Stay' category).
    *   **Categorical Feature Analysis:**
        *   **Action:** Generate bar plots of value counts for each categorical feature.
        *   **Action:** Visualize relationships with 'Stay' (e.g., stacked bar charts of 'Department' vs. 'Stay').
    *   **Age Feature:**
        *   **Action:** Analyze the distribution of 'Age' categories.
        *   **Action:** Convert 'Age' ranges to a numerical representation (e.g., midpoint or ordinal encoding) and analyze its distribution.
    *   **Correlation Analysis:**
        *   **Action:** Compute and visualize a correlation heatmap for numerical features.
    *   **Missing Data Deep Dive:**
        *   **Action:** Re-evaluate missing data patterns after initial profiling. Note any features with high percentages of missing values.
    *   **Tools:** `pandas`, `matplotlib`, `seaborn`.
    *   **Deliverable:** Jupyter notebook with EDA findings, visualizations, and initial hypotheses.

## Phase 2: Advanced Data Preprocessing & Cleaning

1.  **Preprocessing Pipeline Strategy:**
    *   **Action:** Plan to use `sklearn.pipeline.Pipeline` and `ColumnTransformer` to create a reproducible preprocessing workflow.
    *   **Rationale:** Ensures consistent application of transformations to train, validation, and test sets, preventing data leakage.

2.  **Handling Missing Values:**
    *   **For 'Bed Grade' and 'City_Code_Patient':**
        *   **Action (Option 1 - Simple):** Impute with mode (as in original notebook). Ensure mode is calculated from the *training set only*.
        *   **Action (Option 2 - Advanced):**
            *   Use `sklearn.impute.KNNImputer` for 'Bed Grade' (if numerical) or `IterativeImputer`.
            *   For 'City_Code_Patient', consider imputing based on mode within `patientid` groups if feasible, or a global mode from the training set.
        *   **Action:** Integrate the chosen imputer into the `ColumnTransformer`.
    *   **Tool:** `sklearn.impute`.

3.  **Outlier Treatment (for numerical features like `Admission_Deposit`):**
    *   **Action:** Based on EDA, decide on outlier treatment (e.g., capping at 1st and 99th percentiles, log transformation if highly skewed).
    *   **Action:** Implement using `sklearn.preprocessing.FunctionTransformer` or custom functions within the pipeline.
    *   **Note:** Apply outlier treatment fitted *only* on training data.

4.  **Categorical Feature Encoding:**
    *   **'Age' Feature:**
        *   **Action:** Convert to ordinal numerical values (e.g., '0-10' -> 0, '11-20' -> 1, etc.) or use midpoints. Use `OrdinalEncoder` or a custom mapping.
    *   **Nominal Categorical Features (e.g., `Hospital_type_code`, `Department`, `Ward_Type`):**
        *   **Action:** Use `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')`. The `handle_unknown='ignore'` parameter is crucial for handling new categories in the test set.
    *   **High Cardinality Features (e.g., `Hospital_code`, `City_Code_Hospital`, `patientid` if used directly, `City_Code_Patient`):**
        *   **Action (Option 1):** One-Hot Encoding if dimensionality allows.
        *   **Action (Option 2 - Preferred for very high cardinality):** Target Encoding, Weight of Evidence Encoding, or CatBoost Encoder. Implement carefully to prevent data leakage (fit on training folds within cross-validation).
        *   **Tool:** `category_encoders` library (for Target Encoding, etc.), `sklearn.preprocessing.OneHotEncoder`, `OrdinalEncoder`.

5.  **Numerical Feature Scaling:**
    *   **Action:** Apply `sklearn.preprocessing.StandardScaler` or `MinMaxScaler` to all numerical features *after* imputation and outlier handling.
    *   **Action:** Integrate into the `ColumnTransformer`.

6.  **Data Splitting (Initial for Preprocessing Development):**
    *   **Action:** Split the raw training data (`train.csv`) into a temporary training set and a validation set (e.g., 80/20 split) using `train_test_split` with stratification on 'Stay'.
    *   **Rationale:** Develop and test the preprocessing pipeline on this temporary split before moving to full cross-validation.

## Phase 3: Enriched Feature Engineering

*   **Note:** Feature engineering should be done *after* the initial train/validation split to prevent data leakage. New features are derived based on the training portion and then applied to the validation/test set.

1.  **Refined Count/Frequency Features (from original notebook):**
    *   **Action:** Re-implement `get_countid_encode` function.
    *   **Key Improvement:** Ensure that counts for `test` data are derived from aggregations on the *training* data (or use counts from `test` directly if appropriate for the competition/problem but be mindful of leakage). For robust generalization, map training set counts.
    *   **Action:** Apply to `patientid`, `patientid` & `Hospital_region_code`, `patientid` & `Ward_Facility_Code`.
    *   **Action:** Fill NaNs resulting from merges using a global median/mean from the *training set's* engineered feature.

2.  **Interaction Features:**
    *   **Action:** Create interaction terms for features identified as potentially synergistic during EDA. Examples:
        *   `Department` x `Severity of Illness`
        *   `Type of Admission` x `Severity of Illness`
        *   `Hospital_type_code` x `Ward_Type`
    *   **Implementation:** If features are one-hot encoded, interactions can be created by element-wise multiplication or by concatenating and then using a model that can capture interactions (like tree-based models). For categorical features before OHE, concatenate string values and then encode.

3.  **Date/Time Features (If Applicable):**
    *   *Currently, no explicit date/time columns are mentioned other than 'Stay' which is the target. If admission/discharge dates were available, features like day of week, month, or duration calculations would be relevant.*

4.  **Aggregation-based Features:**
    *   **Action:** For features like `patientid` or `Hospital_code`, calculate aggregated statistics from other features on the *training set*.
        *   Example: Average/median `Admission_Deposit` per `Hospital_code`.
        *   Example: Number of previous visits for a `patientid`.
    *   **Action:** Merge these aggregated features back to the main dataset. Handle new `patientid`s or `Hospital_code`s in validation/test (e.g., fill with global average from training).

5.  **Feature Selection (Optional but Recommended):**
    *   **Action:** After creating a broad set of features, consider using feature selection techniques if dimensionality is very high or to remove redundant/noisy features.
    *   **Methods:**
        *   Filter methods (e.g., correlation with target, mutual information).
        *   Wrapper methods (e.g., Recursive Feature Elimination - RFE).
        *   Embedded methods (e.g., L1 regularization, feature importance from tree models).
    *   **Tool:** `sklearn.feature_selection`.

## Phase 4: Model Selection, Training & Optimization

1.  **Experiment Tracking Setup:**
    *   **Action:** Install and configure `mlflow`.
    *   **Action:** Start an MLflow tracking server (can be local).
    *   **Action:** Structure code to log experiments, parameters, metrics, and model artifacts using `mlflow.start_run()`.
    *   **Tool:** `mlflow`.

2.  **Cross-Validation Strategy:**
    *   **Action:** Implement `sklearn.model_selection.StratifiedKFold` (e.g., 5 or 10 folds) on the fully preprocessed training data.
    *   **Rationale:** Provides a robust estimate of model performance and helps in hyperparameter tuning.

3.  **Model Selection & Baseline:**
    *   **Action:** Choose a diverse set of models:
        *   **Tree-based (High Priority):** `xgboost.XGBClassifier`, `lightgbm.LGBMClassifier`, `catboost.CatBoostClassifier`.
        *   **Neural Network (High Priority):** Using Keras/TensorFlow or PyTorch. Correct the input layer shape (`input_shape=(num_features,)`). Experiment with architectures (number of layers, neurons, activations like ReLU, Tanh). Add Dropout and Batch Normalization.
        *   **Others for Comparison:** `sklearn.ensemble.RandomForestClassifier`, `sklearn.linear_model.LogisticRegression` (as a baseline).
    *   **Action:** For each model, train and evaluate using the cross-validation setup. Log initial (default hyperparameter) performance.

4.  **Hyperparameter Tuning:**
    *   **Action:** For the most promising models (XGBoost, LightGBM, CatBoost, Neural Network), perform hyperparameter optimization.
    *   **Tools & Techniques:**
        *   `sklearn.model_selection.RandomizedSearchCV` or `GridSearchCV` with `cv=StratifiedKFold_object`.
        *   Bayesian Optimization libraries: `hyperopt`, `optuna`, `scikit-optimize`.
    *   **Action:** Define a hyperparameter search space for each model.
    *   **Action:** Select an appropriate scoring metric for tuning (e.g., `f1_weighted` or `f1_macro`).
    *   **Action:** Log all tuning experiments and best parameters in MLflow.

5.  **Addressing Class Imbalance (within CV loop):**
    *   **Technique 1: Class Weights:**
        *   **Action:** For models that support it (XGBoost, LightGBM, Keras), use the `class_weight` or `scale_pos_weight` (for binary, adapt for multiclass) parameter. Calculate weights inversely proportional to class frequencies from the *training fold*.
    *   **Technique 2: Resampling (on training folds):**
        *   **Action:** Use `imblearn.over_sampling.SMOTE` or `imblearn.under_sampling.RandomUnderSampler`.
        *   **Important:** Apply resampling *only* to the training portion of each fold in the cross-validation loop to prevent data leakage into the validation fold.
        *   **Tool:** `imblearn.pipeline.Pipeline` to integrate resampling with the model.

## Phase 5: Rigorous Model Evaluation

1.  **Define Primary and Secondary Metrics:**
    *   **Primary:** `f1_weighted` (good overall for imbalanced multi-class).
    *   **Secondary:** `f1_macro` (if equal performance across classes is important), `accuracy`, `roc_auc_ovr_weighted` (if probabilities are well-calibrated).
    *   **Competition Specific:** If this were a Kaggle competition, use their specified metric.

2.  **Evaluate Best Models from Hyperparameter Tuning:**
    *   **Action:** Use the best hyperparameters found and evaluate on the chosen cross-validation strategy.
    *   **Action:** Log final cross-validated scores for all metrics.

3.  **Confusion Matrix Analysis:**
    *   **Action:** Generate and visualize the confusion matrix (summed over CV folds or on a hold-out validation set).
    *   **Action:** Analyze per-class precision, recall, and F1-score. Identify which 'Stay' categories are predicted well and which are challenging.

4.  **Model Interpretability:**
    *   **Tree-based Models:**
        *   **Action:** Plot feature importances (Gini importance, permutation importance).
    *   **All Models (SHAP):**
        *   **Action:** Install `shap`.
        *   **Action:** Use SHAP to generate summary plots and dependence plots to understand feature contributions globally and for specific predictions.
        *   **Deliverable:** Visualizations and insights on feature importance.

5.  **Error Analysis:**
    *   **Action:** Examine instances where the best model makes significant errors. Try to identify patterns or feature characteristics associated with misclassifications.

## Phase 6: Prediction, Final Output & Reporting

1.  **Train Final Model:**
    *   **Action:** Select the overall best model and hyperparameter combination based on robust CV evaluation.
    *   **Action:** Train this final model on the *entire* preprocessed training dataset (`train.csv` data).

2.  **Preprocess Test Data:**
    *   **Action:** Apply the *exact same* preprocessing pipeline (fitted on the full training data) to the `test.csv` data.

3.  **Generate Predictions on Test Set:**
    *   **Action:** Use the trained final model to predict 'Stay' categories for the preprocessed test data.
    *   **Action:** For Neural Networks, use `np.argmax(model.predict(test_data_processed), axis=-1)`.
    *   **Action:** Inverse transform the numerical predictions back to their original string labels (e.g., using `label_encoder.inverse_transform()`).

4.  **Create Submission File:**
    *   **Action:** Format the predictions into a CSV file with `case_id` and the predicted 'Stay' string.
    *   **Deliverable:** `submission.csv`.

5.  **Project Report & Documentation:**
    *   **Action:** Summarize the entire process, from EDA to final model, including:
        *   Key EDA insights.
        *   Preprocessing steps and rationale.
        *   Feature engineering choices.
        *   Model selection process and hyperparameter tuning results.
        *   Final model performance and evaluation.
        *   Interpretability findings.
        *   Challenges encountered and potential future improvements.
    *   **Deliverable:** A comprehensive project report (Markdown or PDF).

## Phase 7: Potential Future Enhancements & Deployment Considerations

1.  **Model Ensembling/Stacking:**
    *   **Action:** Combine predictions from multiple strong, diverse models (e.g., XGBoost, LightGBM, NN) using techniques like averaging, voting, or stacking generalization.

2.  **Advanced Feature Engineering (Iterative):**
    *   Explore more complex features, e.g., embeddings for high-cardinality categorical features if not already done, or domain-specific features if expert knowledge is available.

3.  **Model Deployment Strategy:**
    *   **Action (Conceptual):** Outline how the model (including the preprocessing pipeline) could be deployed.
        *   Save model and pipeline: `joblib.dump()` or `mlflow.sklearn.log_model()`.
        *   Options: REST API (Flask/FastAPI), batch prediction script, serverless function.

4.  **Monitoring and Maintenance Plan:**
    *   **Action (Conceptual):** Discuss how to monitor model performance in a production environment (e.g., tracking prediction drift, accuracy against new ground truth if it becomes available, data drift detection).
    *   **Action (Conceptual):** Plan for model retraining schedules.

This detailed plan provides a roadmap. Adjustments may be needed based on findings at each step. 