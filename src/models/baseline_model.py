import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from typing import Any, Dict, List, Tuple, Union

from src.models.base_model import BaseModel
from src.data.preprocessing import apply_age_ordinal_mapping, get_feature_types, get_preprocessor
# from src.data.make_dataset import load_data # Placeholder, will be implemented in T004
# from src.evaluation.cross_validation import create_stratified_kfold # Placeholder, will be implemented in T006

# Setup MLflow
experiment_name = "Healthcare_LOS_Prediction"
mlflow.set_experiment(experiment_name)

class BaselineModel(BaseModel):
    """
    A baseline Logistic Regression model for Healthcare Length of Stay Prediction.

    This model inherits from BaseModel and implements its abstract methods for
    training, prediction, and evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the BaselineModel with a Logistic Regression classifier
        and a preprocessing pipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
                                      Expected keys: 'model_params' for LogisticRegression.
        """
        super().__init__(config)
        self.model_params = self.config.get('model_params', {})
        self.classifier = LogisticRegression(**self.model_params)
        self.preprocessor_config = self.config.get('preprocessing_params', {})
        self.preprocessor = None # Will be fitted during training

        self.pipeline = Pipeline(steps=[
            ('preprocessor', None), # Placeholder, will be set in train
            ('classifier', self.classifier)
        ])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Trains the Logistic Regression model, including fitting the preprocessor.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            **kwargs: Additional keyword arguments (e.g., 'fold_idx' for MLflow logging).
        """
        fold_idx = kwargs.get('fold_idx', None)
        
        # Apply age ordinal mapping
        X_train_processed_age = apply_age_ordinal_mapping(X_train)
        
        # Identify feature types
        numeric_features, categorical_features = get_feature_types(X_train_processed_age)
        
        # Get and fit the preprocessor
        self.preprocessor = get_preprocessor(numeric_features, categorical_features, self.preprocessor_config)
        
        # Update the pipeline with the fitted preprocessor
        self.pipeline.set_params(preprocessor=self.preprocessor)

        print(f"Training BaselineModel with config: {self.config}")
        self.pipeline.fit(X_train_processed_age, y_train)
        print("BaselineModel training complete.")

        # MLflow logging for parameters (only once per run, not per fold)
        if fold_idx is None or fold_idx == 0: # Log params only for the first fold or if not in CV
            mlflow.log_params({
                "model_type": "LogisticRegression",
                **self.model_params
            })
            print(f"Logged model parameters to MLflow: {self.model_params}")

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """
        Makes predictions using the trained Logistic Regression model.

        Args:
            X (pd.DataFrame): Input features for prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: Predicted labels.
        """
        if self.pipeline is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor has not been trained. Call train() first.")
        
        print("Making predictions with BaselineModel...")
        X_processed_age = apply_age_ordinal_mapping(X)
        predictions = self.pipeline.predict(X_processed_age)
        return pd.Series(predictions, index=X.index)

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, **kwargs: Any) -> Dict[str, float]:
        """
        Evaluates the model's predictions and returns a dictionary of metrics.

        Args:
            y_true (pd.Series): True labels.
            y_pred (pd.Series): Predicted labels.
            **kwargs: Additional keyword arguments (e.g., 'fold_idx' for MLflow logging).

        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        }
        
        fold_idx = kwargs.get('fold_idx', None)
        if fold_idx is not None:
            mlflow.log_metric(f"fold_{fold_idx+1}_accuracy", accuracy)
            mlflow.log_metric(f"fold_{fold_idx+1}_f1_weighted", f1_weighted)
            mlflow.log_metric(f"fold_{fold_idx+1}_f1_macro", f1_macro)
            print(f"Fold {fold_idx+1} - Accuracy: {accuracy:.4f}, "
                  f"F1 (weighted): {f1_weighted:.4f}, "
                  f"F1 (macro): {f1_macro:.4f}")
        else:
            print(f"Evaluation - Accuracy: {accuracy:.4f}, "
                  f"F1 (weighted): {f1_weighted:.4f}, "
                  f"F1 (macro): {f1_macro:.4f}")
            
        return metrics

def run_baseline_evaluation_refactored():
    """
    Orchestrates the complete baseline model evaluation process using the refactored class.
    This function will simulate data loading and cross-validation until dedicated modules are ready.
    """
    print("Running refactored baseline evaluation...")

    # --- Simulate Data Loading (replace with actual load_data from src.data.make_dataset later) ---
    # For demonstration, create dummy data
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=11, random_state=42)
    X_dummy_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(20)])
    y_dummy_series = pd.Series(y_dummy)

    # Add some categorical features and 'Age' for testing preprocessor
    X_dummy_df['categorical_feature_1'] = np.random.choice(['A', 'B', 'C'], size=1000)
    X_dummy_df['categorical_feature_2'] = np.random.choice(['X', 'Y'], size=1000)
    age_categories = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', 'More than 100 Days']
    X_dummy_df['Age'] = np.random.choice(age_categories, size=1000)
    
    X_data = X_dummy_df
    y_data = y_dummy_series
    print("Simulated data loaded.")

    # --- Simulate Cross-Validation Folds (replace with actual create_stratified_kfold later) ---
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = list(skf.split(X_data, y_data))
    print(f"Simulated {len(cv_folds)} cross-validation folds.")

    # Model configuration
    model_config = {
        'model_params': {
            'solver': 'liblinear',
            'C': 1.0,
            'max_iter': 1000,
            'multi_class': 'ovr',
            'random_state': 42
        }
    }

    # Initialize the refactored baseline model
    baseline_model = BaselineModel(config=model_config)
    
    all_y_test = []
    all_y_pred = []
    
    # Start MLflow run
    with mlflow.start_run(run_name="Refactored_Baseline_LogisticRegression"):
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
            print(f"\nTraining on fold {fold_idx+1}/{len(cv_folds)}")
            
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            # Train the model for the current fold
            baseline_model.train(X_train, y_train, fold_idx=fold_idx)
            
            # Make predictions
            y_pred = baseline_model.predict(X_test)
            
            # Evaluate and log metrics for the fold
            metrics = baseline_model.evaluate(y_test, y_pred, fold_idx=fold_idx)
            
            all_y_test.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
        
        # Calculate and log average metrics
        # This part needs to be refactored into a dedicated evaluation module (T006)
        # For now, we'll just log the overall metrics from the last fold or aggregate manually.
        # A more robust aggregation would collect metrics from each fold and average them.
        # For simplicity, let's re-evaluate on the concatenated predictions for overall report.
        overall_accuracy = accuracy_score(all_y_test, all_y_pred)
        overall_f1_weighted = f1_score(all_y_test, all_y_pred, average='weighted')
        overall_f1_macro = f1_score(all_y_test, all_y_pred, average='macro')
        
        mlflow.log_metric("overall_accuracy", overall_accuracy)
        mlflow.log_metric("overall_f1_weighted", overall_f1_weighted)
        mlflow.log_metric("overall_f1_macro", overall_f1_macro)
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}, "
              f"Overall F1 (weighted): {overall_f1_weighted:.4f}, "
              f"Overall F1 (macro): {overall_f1_macro:.4f}")

        # Generate and log confusion matrix
        labels = sorted(y_data.unique())
        cm = confusion_matrix(all_y_test, all_y_pred, labels=labels)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Refactored Baseline Logistic Regression')
        plt.tight_layout()
        plt.savefig('confusion_matrix_refactored_baseline.png')
        mlflow.log_artifact('confusion_matrix_refactored_baseline.png')
        
        # Log classification report
        report = classification_report(all_y_test, all_y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report_refactored_baseline.csv')
        mlflow.log_artifact('classification_report_refactored_baseline.csv')
        
        # Log the model
        mlflow.sklearn.log_model(baseline_model.pipeline, "refactored_baseline_logistic_regression_model")
        
        print("\nRefactored baseline model evaluation completed and logged to MLflow!")

if __name__ == "__main__":
    run_baseline_evaluation_refactored()
