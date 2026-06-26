import os
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.sparse
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.xgboost
from typing import Any, Dict, List, Tuple, Union

from src.models.base_model import BaseModel
from src.data.preprocessing import apply_age_ordinal_mapping, get_feature_types, get_preprocessor, inverse_transform_predictions
# from src.data.make_dataset import load_data # Placeholder, will be implemented in T004
# from src.evaluation.cross_validation import create_stratified_kfold # Placeholder, will be implemented in T006

# Setup MLflow
experiment_name = "Healthcare_LOS_Prediction"
mlflow.set_experiment(experiment_name)

class XGBoostModel(BaseModel):
    """
    An XGBoost model for Healthcare Length of Stay Prediction.

    This model inherits from BaseModel and implements its abstract methods for
    training, prediction, and evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the XGBoostModel with an XGBoost classifier
        and a preprocessing pipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
                                      Expected keys: 'model_params' for XGBoost,
                                      'label_encoder_classes' if pre-fitted.
        """
        super().__init__(config)
        self.model_params = self.config.get('model_params', {})
        self.num_boost_round = self.config.get('num_boost_round', 100)
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', 10)
        self.preprocessor_config = self.config.get('preprocessing_params', {})
        
        self.label_encoder = LabelEncoder()
        if 'label_encoder_classes' in self.config:
            self.label_encoder.classes_ = np.array(self.config['label_encoder_classes'])

        self.preprocessor = None # Will be fitted during training
        self.model = None # XGBoost trained model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Trains the XGBoost model, including fitting the preprocessor and label encoder.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            **kwargs: Additional keyword arguments (e.g., 'fold_idx' for MLflow logging).
        """
        fold_idx = kwargs.get('fold_idx', None)
        
        # Apply age ordinal mapping and identify feature types
        X_train_processed_age = apply_age_ordinal_mapping(X_train)
        numeric_features, categorical_features = get_feature_types(X_train_processed_age)
        
        # Fit label encoder
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        if 'label_encoder_classes' not in self.config:
            self.config['label_encoder_classes'] = self.label_encoder.classes_.tolist()
            print("Target Class Mapping:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                print(f"  {class_name}: {i}")

        # Get and fit the preprocessor
        self.preprocessor = get_preprocessor(numeric_features, categorical_features, self.preprocessor_config)
        X_train_transformed = self.preprocessor.fit_transform(X_train_processed_age)
        
        # Convert to dense array if sparse
        if scipy.sparse.issparse(X_train_transformed):
            X_train_transformed = X_train_transformed.toarray()
        
        # Convert to DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_train_transformed, label=y_train_encoded)
        
        print(f"Training XGBoostModel with config: {self.config}")
        
        # Train the model
        self.model = xgb.train(
            self.model_params,
            dtrain,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        print("XGBoostModel training complete.")

        # MLflow logging for parameters (only once per run, not per fold)
        if fold_idx is None or fold_idx == 0: # Log params only for the first fold or if not in CV
            mlflow.log_params({
                "model_type": "XGBoost",
                "num_boost_round": self.num_boost_round,
                "early_stopping_rounds": self.early_stopping_rounds,
                **self.model_params
            })
            print(f"Logged model parameters to MLflow: {self.model_params}")

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """
        Makes predictions using the trained XGBoost model.

        Args:
            X (pd.DataFrame): Input features for prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: Predicted labels.
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor has not been trained. Call train() first.")
        
        print("Making predictions with XGBoostModel...")
        X_processed_age = apply_age_ordinal_mapping(X)
        X_transformed = self.preprocessor.transform(X_processed_age)

        if scipy.sparse.issparse(X_transformed):
            X_transformed = X_transformed.toarray()

        dtest = xgb.DMatrix(X_transformed)
        y_pred_probs = self.model.predict(dtest)
        y_pred_encoded = np.argmax(y_pred_probs, axis=1)
        
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
                      'log_artifacts' for logging confusion matrix/report).

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
            plt.title('Confusion Matrix - XGBoost')
            plt.tight_layout()
            cm_path = 'confusion_matrix_xgboost.png'
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close() # Close plot to free memory
            
            # Log classification report
            report = classification_report(y_true_encoded, y_pred_encoded, output_dict=True, target_names=self.label_encoder.classes_)
            report_df = pd.DataFrame(report).transpose()
            report_path = 'classification_report_xgboost.csv'
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)

            # Log feature importance (requires trained model)
            if self.model:
                importance = self.model.get_score(importance_type='gain')
                
                # Get feature names from the preprocessor
                feature_names_out = self.preprocessor.get_feature_names_out()
                
                # Map feature names back if possible, otherwise use generic names
                mapped_importance = {}
                for k, v in importance.items():
                    try:
                        # Attempt to map 'fX' to actual feature names
                        idx = int(k.replace('f', ''))
                        if idx < len(feature_names_out):
                            mapped_importance[feature_names_out[idx]] = v
                        else:
                            mapped_importance[k] = v # Fallback if index out of bounds
                    except (ValueError, AttributeError):
                        mapped_importance[k] = v # Fallback for non-'fX' keys
                
                sorted_importances = dict(sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True))
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(sorted_importances)), list(sorted_importances.values()))
                plt.yticks(range(len(sorted_importances)), list(sorted_importances.keys()))
                plt.xlabel('Importance')
                plt.title('XGBoost Feature Importance (Gain)')
                plt.tight_layout()
                importance_path = 'feature_importance_xgboost.png'
                plt.savefig(importance_path)
                mlflow.log_artifact(importance_path)
                plt.close() # Close plot to free memory
                
                importance_df = pd.DataFrame({
                    'Feature': list(sorted_importances.keys()),
                    'Importance': list(sorted_importances.values())
                })
                importance_csv = 'feature_importance_xgboost.csv'
                importance_df.to_csv(importance_csv, index=False)
                mlflow.log_artifact(importance_csv)
            
        return metrics

def run_xgboost_evaluation_refactored():
    """
    Orchestrates the complete XGBoost model evaluation process using the refactored class.
    This function will simulate data loading and cross-validation until dedicated modules are ready.
    """
    print("Running refactored XGBoost evaluation...")

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

    # Model configuration
    model_config = {
        'model_params': {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_data)), # Dynamically set num_class
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'tree_method': 'hist',
            'verbosity': 0
        },
        'num_boost_round': 100,
        'early_stopping_rounds': 10
    }

    # Initialize the refactored XGBoost model
    xgboost_model = XGBoostModel(config=model_config)
    
    all_y_test = []
    all_y_pred = []
    
    # Start MLflow run
    with mlflow.start_run(run_name="Refactored_XGBoost_Model"):
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
            print(f"\nTraining on fold {fold_idx+1}/{len(cv_folds)}")
            
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            # Train the model for the current fold
            xgboost_model.train(X_train, y_train, fold_idx=fold_idx)
            
            # Make predictions
            y_pred = xgboost_model.predict(X_test)
            
            # Evaluate and log metrics for the fold
            metrics = xgboost_model.evaluate(y_test, y_pred, fold_idx=fold_idx)
            
            all_y_test.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
        
        # Calculate and log overall metrics and artifacts
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

        # Log overall artifacts using the evaluate method with log_artifacts=True
        # Need to pass the original y_data to evaluate for correct label encoding in plotting
        xgboost_model.evaluate(pd.Series(all_y_test), pd.Series(all_y_pred), log_artifacts=True)
        
        # Log the model (the last trained model from the CV loop)
        mlflow.sklearn.log_model(xgboost_model.model, "refactored_xgboost_model")
        
        print("\nRefactored XGBoost model evaluation completed and logged to MLflow!")

if __name__ == "__main__":
    run_xgboost_evaluation_refactored()
