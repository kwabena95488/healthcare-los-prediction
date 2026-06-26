import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.catboost
import shap # For interpretability
import time
from typing import Any, Dict, List, Tuple, Union

from src.models.base_model import BaseModel
from src.data.preprocessing import apply_age_ordinal_mapping, get_feature_types, inverse_transform_predictions
# from src.data.make_dataset import load_data # Placeholder, will be implemented in T004
# from src.evaluation.cross_validation import create_stratified_kfold # Placeholder, will be implemented in T006

# Set up MLflow
experiment_name = "Healthcare_LOS_Prediction"
mlflow.set_experiment(experiment_name)

class CatBoostModel(BaseModel):
    """
    A CatBoost model for Healthcare Length of Stay Prediction.

    This model inherits from BaseModel and implements its abstract methods for
    training, prediction, and evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the CatBoostModel with CatBoost parameters.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
                                      Expected keys: 'model_params' for CatBoost,
                                      'label_encoder_classes' if pre-fitted.
        """
        super().__init__(config)
        self.model_params = self.config.get('model_params', {})
        
        self.label_encoder = LabelEncoder()
        if 'label_encoder_classes' in self.config:
            self.label_encoder.classes_ = np.array(self.config['label_encoder_classes'])

        self.model = None # CatBoost trained model
        self.categorical_features_names = [] # To store names of categorical features
        self.preprocessor_config = self.config.get('preprocessing_params', {})

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Trains the CatBoost model, including fitting the label encoder.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            **kwargs: Additional keyword arguments (e.g., 'fold_idx' for MLflow logging).
        """
        fold_idx = kwargs.get('fold_idx', None)
        
        # Apply age ordinal mapping and identify feature types
        X_train_processed_age = apply_age_ordinal_mapping(X_train)
        _, categorical_features_names = get_feature_types(X_train_processed_age)
        categorical_indices = [X_train_processed_age.columns.get_loc(col) for col in categorical_features_names]
        self.categorical_features_names = categorical_features_names # Store for later use
        
        # Fit label encoder
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        if 'label_encoder_classes' not in self.config:
            self.config['label_encoder_classes'] = self.label_encoder.classes_.tolist()
            print("Target Class Mapping:", self.label_encoder.classes_)

        print(f"Training CatBoostModel with config: {self.config}")
        
        # Create CatBoost Pool
        train_pool = Pool(X_train_processed_age, y_train_encoded, cat_features=categorical_indices)
        
        # Initialize and train the model
        self.model = CatBoostClassifier(**self.model_params)
        self.model.fit(train_pool, verbose=False) # Removed eval_set for simplicity in train method
        
        print("CatBoostModel training complete.")

        # MLflow logging for parameters (only once per run, not per fold)
        if fold_idx is None or fold_idx == 0: # Log params only for the first fold or if not in CV
            mlflow.log_params({
                "model_type": "CatBoost",
                **self.model_params
            })
            print(f"Logged model parameters to MLflow: {self.model_params}")

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """
        Makes predictions using the trained CatBoost model.

        Args:
            X (pd.DataFrame): Input features for prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: Predicted labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        print("Making predictions with CatBoostModel...")
        X_processed_age = apply_age_ordinal_mapping(X)
        _, categorical_features_names = get_feature_types(X_processed_age)
        categorical_indices = [X_processed_age.columns.get_loc(col) for col in categorical_features_names]
        
        # Create CatBoost Pool for prediction
        predict_pool = Pool(X_processed_age, cat_features=categorical_indices)
        
        y_pred_encoded = self.model.predict(predict_pool)
        
        # CatBoost predict returns a 2D array for multiclass, need to flatten
        if y_pred_encoded.ndim > 1:
            y_pred_encoded = y_pred_encoded.flatten()
        
        # Inverse transform predictions to original labels
        y_pred = inverse_transform_predictions(y_pred_encoded, self.label_encoder)
        return pd.Series(y_pred, index=X.index)

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, **kwargs: Any) -> Dict[str, float]:
        """
        Evaluates the model's predictions and returns a dictionary of metrics.

        Args:
            y_true (pd.Series): True labels.
            y_pred (pd.Series): Predicted labels.
            **kwargs: Additional keyword arguments (e.g., 'fold_idx' for MLflow logging,
                      'log_artifacts' for logging confusion matrix/report, 'X_test' for SHAP).

        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        # Ensure y_true is encoded for metric calculation if it's not already
        y_true_encoded = self.label_encoder.transform(y_true)
        y_pred_encoded = self.label_encoder.transform(y_pred) # Ensure y_pred is also encoded

        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
        f1_macro = f1_score(y_true_encoded, y_pred_encoded, average='macro')

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
            
        log_artifacts = kwargs.get('log_artifacts', False)
        if log_artifacts:
            # Generate and log confusion matrix
            labels = sorted(y_true.unique())
            cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=self.label_encoder.transform(labels))
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix - CatBoost')
            plt.tight_layout()
            cm_path = 'confusion_matrix_catboost.png'
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close() # Close plot to free memory
            
            # Log classification report
            report = classification_report(y_true_encoded, y_pred_encoded, output_dict=True, target_names=self.label_encoder.classes_)
            report_df = pd.DataFrame(report).transpose()
            report_path = 'classification_report_catboost.csv'
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)

            # Log feature importance (requires trained model)
            if self.model:
                feature_importances = self.model.feature_importances_
                feature_names = self.model.feature_names_
                
                if feature_names and len(feature_importances) == len(feature_names):
                    sorted_importances = dict(sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True))
                else:
                    print("Warning: Feature importance length mismatch or names not available. Logging raw importance.")
                    sorted_importances = {f"feature_{i}": imp for i, imp in enumerate(feature_importances)}
                    sorted_importances = dict(sorted(sorted_importances.items(), key=lambda x: x[1], reverse=True))

                plt.figure(figsize=(12, 8))
                plt.barh(range(len(sorted_importances)), list(sorted_importances.values()))
                plt.yticks(range(len(sorted_importances)), list(sorted_importances.keys()))
                plt.xlabel('Importance')
                plt.title('CatBoost Feature Importance')
                plt.tight_layout()
                importance_path = 'feature_importance_catboost.png'
                plt.savefig(importance_path)
                mlflow.log_artifact(importance_path)
                plt.close() # Close plot to free memory
                
                importance_df = pd.DataFrame({
                    'Feature': list(sorted_importances.keys()),
                    'Importance': list(sorted_importances.values())
                })
                importance_csv = 'feature_importance_catboost.csv'
                importance_df.to_csv(importance_csv, index=False)
                mlflow.log_artifact(importance_csv)
            
            # Try to generate SHAP values (on a sample for speed)
            X_test_for_shap = kwargs.get('X_test_for_shap', None)
            if X_test_for_shap is not None and self.model:
                try:
                    # Sample data for SHAP (for speed)
                    sample_size = min(500, len(X_test_for_shap))
                    X_sample = X_test_for_shap.sample(sample_size, random_state=42)
                    
                    # Prepare X_sample for SHAP explainer (handle Age ordinal encoding)
                    X_sample_processed_age = apply_age_ordinal_mapping(X_sample)
                    _, categorical_features_names_shap = get_feature_types(X_sample_processed_age)
                    categorical_indices_shap = [X_sample_processed_age.columns.get_loc(col) for col in categorical_features_names_shap]
                    
                    # Create explainer
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(X_sample_processed_age, cat_features=categorical_indices_shap)
                    
                    # Plot SHAP summary
                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(shap_values, X_sample_processed_age, plot_type="bar", show=False)
                    plt.title("SHAP Feature Importance")
                    plt.tight_layout()
                    shap_file = 'catboost_shap_importance.png'
                    plt.savefig(shap_file)
                    mlflow.log_artifact(shap_file)
                    plt.close()
                except Exception as e:
                    print(f"Could not generate SHAP values: {e}")
            
        return metrics

def run_catboost_evaluation_refactored():
    """
    Orchestrates the complete CatBoost model evaluation process using the refactored class.
    This function will simulate data loading and cross-validation until dedicated modules are ready.
    """
    print("Running refactored CatBoost evaluation...")

    # --- Simulate Data Loading (replace with actual load_data from src.data.make_dataset later) ---
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

    # Model configuration (using fixed parameters for now, assuming best_params from Optuna)
    model_config = {
        'model_params': {
            'loss_function': 'MultiClass',
            'custom_loss': ['Accuracy', 'F1'],
            'eval_metric': 'Accuracy',
            'iterations': 100, # Example fixed iteration
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 20,
            'verbose': False,
            'thread_count': -1,
            'num_class': len(np.unique(y_data)) # Dynamically set num_class
        }
    }

    # Initialize the refactored CatBoost model
    catboost_model = CatBoostModel(config=model_config)
    
    all_y_test = []
    all_y_pred = []
    
    # Start MLflow run
    with mlflow.start_run(run_name="Refactored_CatBoost_Model"):
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
            print(f"\nTraining on fold {fold_idx+1}/{len(cv_folds)}")
            
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            # Train the model for the current fold
            catboost_model.train(X_train, y_train, fold_idx=fold_idx)
            
            # Make predictions
            y_pred = catboost_model.predict(X_test)
            
            # Evaluate and log metrics for the fold
            metrics = catboost_model.evaluate(y_test, y_pred, fold_idx=fold_idx, X_test_for_shap=X_test) # Pass X_test for SHAP
            
            all_y_test.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
        
        # Calculate and log overall metrics and artifacts
        overall_accuracy = accuracy_score(all_y_test, all_y_pred)
        overall_f1_weighted = f1_score(all_y_test, all_y_pred, average='weighted')
        overall_f1_macro = f1_score(all_y_test, all_y_pred, average='macro')
        
        mlflow.log_metric("overall_accuracy", overall_accuracy)
        mlflow.log_metric("overall_f1_weighted", overall_f1_weighted)
        mlflow.log_metric("overall_f1_macro", overall_f1_macro)
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}, "
              f"Overall F1 (weighted): {overall_f1_weighted:.4f}, "
              f"Overall F1 (macro): {overall_f1_macro:.4f}")

        # Log overall artifacts using the evaluate method with log_artifacts=True
        catboost_model.evaluate(pd.Series(all_y_test), pd.Series(all_y_pred), log_artifacts=True, X_test_for_shap=X_data) # Pass X_data for overall SHAP
        
        # Log the model (the last trained model from the CV loop)
        mlflow.catboost.log_model(catboost_model.model, "refactored_catboost_model")
        
        print("\nRefactored CatBoost model evaluation completed and logged to MLflow!")

if __name__ == "__main__":
    run_catboost_evaluation_refactored()
