import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import mlflow
import mlflow.sklearn
import time
import joblib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from src.models.base_model import BaseModel
from src.models.baseline_model import BaselineModel
from src.models.gradient_boosting.xgboost_model import XGBoostModel
from src.models.gradient_boosting.lightgbm_model import LightGBMModel
from src.models.gradient_boosting.catboost_model import CatBoostModel
from src.models.neural_network_model import NeuralNetworkModel

# from src.data.make_dataset import load_data # Placeholder, will be implemented in T004
# from src.evaluation.cross_validation import create_stratified_kfold # Placeholder, will be implemented in T006

# Set up MLflow
experiment_name = "Healthcare_LOS_Prediction"
mlflow.set_experiment(experiment_name)

class EnsembleModel(BaseModel):
    """
    An Ensemble model for Healthcare Length of Stay Prediction.

    This model combines predictions from multiple base models (Logistic Regression,
    XGBoost, LightGBM, CatBoost, Neural Network) using a VotingClassifier.
    It inherits from BaseModel and implements its abstract methods for
    training, prediction, and evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the EnsembleModel by loading and configuring its base models.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the ensemble.
                                      Expected keys: 'base_models_config' (dict of configs for each base model),
                                      'voting_params' for VotingClassifier.
        """
        super().__init__(config)
        self.base_models_config = self.config.get('base_models_config', {})
        self.voting_params = self.config.get('voting_params', {'voting': 'soft', 'n_jobs': -1})
        
        self.base_models = self._load_base_models()
        self.ensemble_classifier = None # VotingClassifier

        # LabelEncoder will be fitted by the first base model (e.g., BaselineModel)
        # and then shared or re-fitted as needed. For now, assume it's handled by base models.
        self.label_encoder = LabelEncoder()
        if 'label_encoder_classes' in self.config:
            self.label_encoder.classes_ = np.array(self.config['label_encoder_classes'])

    def _load_base_models(self) -> Dict[str, BaseModel]:
        """
        Instantiates base models based on the provided configuration.
        """
        models = {}
        for model_name, model_config in self.base_models_config.items():
            if model_name == 'baseline':
                models[model_name] = BaselineModel(config=model_config)
            elif model_name == 'xgboost':
                models[model_name] = XGBoostModel(config=model_config)
            elif model_name == 'lightgbm':
                models[model_name] = LightGBMModel(config=model_config)
            elif model_name == 'catboost':
                models[model_name] = CatBoostModel(config=model_config)
            elif model_name == 'neural_network':
                models[model_name] = NeuralNetworkModel(config=model_config)
            else:
                raise ValueError(f"Unknown base model type: {model_name}")
        return models

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Trains each base model and then fits the VotingClassifier.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            **kwargs: Additional keyword arguments (e.g., 'fold_idx' for MLflow logging).
        """
        fold_idx = kwargs.get('fold_idx', None)
        
        print(f"Training EnsembleModel with config: {self.config}")
        
        # Fit label encoder for the ensemble (if not already fitted by a base model)
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y_train)
            self.config['label_encoder_classes'] = self.label_encoder.classes_.tolist()
            print("Ensemble LabelEncoder fitted.")

        # Train individual base models
        estimators = []
        for name, model_instance in self.base_models.items():
            print(f"Training base model: {name}...")
            # Pass label encoder classes to base models if they need it for consistency
            if 'label_encoder_classes' not in model_instance.config:
                model_instance.config['label_encoder_classes'] = self.label_encoder.classes_.tolist()
            
            model_instance.train(X_train, y_train, fold_idx=fold_idx)
            
            # For VotingClassifier, we need the actual sklearn model/pipeline, not our BaseModel wrapper
            # Ensure the estimator can handle raw X_train by including its preprocessor
            if hasattr(model_instance, 'pipeline') and model_instance.pipeline is not None: # Baseline, NeuralNetwork
                estimators.append((name, model_instance.pipeline))
            elif hasattr(model_instance, 'model') and model_instance.model is not None and hasattr(model_instance, 'preprocessor') and model_instance.preprocessor is not None:
                # For XGBoost, LightGBM, CatBoost, wrap their preprocessor and model in a new pipeline
                from sklearn.pipeline import Pipeline as SklearnPipeline # Avoid name collision
                pipeline_for_ensemble = SklearnPipeline(steps=[
                    ('preprocessor', model_instance.preprocessor),
                    ('classifier', model_instance.model)
                ])
                estimators.append((name, pipeline_for_ensemble))
            else:
                raise RuntimeError(f"Base model {name} did not expose a scikit-learn compatible model/pipeline or preprocessor.")

        # Create and train the ensemble classifier
        self.ensemble_classifier = VotingClassifier(estimators=estimators, **self.voting_params)
        
        # The VotingClassifier's fit method will internally pass X_train to each estimator's fit method.
        # Since our estimators (pipelines) now handle preprocessing, we pass raw X_train.
        # Target y_train needs to be label encoded.
        y_train_encoded = self.label_encoder.transform(y_train)
        self.ensemble_classifier.fit(X_train, y_train_encoded)

        print("EnsembleModel training complete.")

        # MLflow logging for parameters (only once per run, not per fold)
        if fold_idx is None or fold_idx == 0: # Log params only for the first fold or if not in CV
            mlflow.log_params({
                "model_type": "Ensemble",
                "base_models": list(self.base_models.keys()),
                **self.voting_params
            })
            print(f"Logged ensemble parameters to MLflow: {self.voting_params}")

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """
        Makes predictions using the trained Ensemble model.

        Args:
            X (pd.DataFrame): Input features for prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: Predicted labels.
        """
        if self.ensemble_classifier is None:
            raise RuntimeError("Ensemble model has not been trained. Call train() first.")
        
        print("Making predictions with EnsembleModel...")
        # The VotingClassifier expects preprocessed data if its estimators are pipelines.
        # However, our base models handle their own preprocessing.
        # This means we need to ensure the VotingClassifier's estimators are correctly
        # set up to receive raw data and preprocess it internally, or we preprocess here.
        # For simplicity, we'll assume the estimators passed to VotingClassifier (model_instance.pipeline/model)
        # are already capable of handling raw X.
        
        # The VotingClassifier's predict method will call predict on its internal estimators.
        # These estimators are the actual sklearn pipelines/models from base_models.
        # So, we pass the raw X, and the internal pipelines/models will handle preprocessing.
        y_pred_encoded = self.ensemble_classifier.predict(X)
        
        # Inverse transform predictions to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
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
            plt.title('Confusion Matrix - Ensemble Model')
            plt.tight_layout()
            cm_path = 'confusion_matrix_ensemble.png'
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close() # Close plot to free memory
            
            # Log classification report
            report = classification_report(y_true_encoded, y_pred_encoded, output_dict=True, target_names=self.label_encoder.classes_)
            report_df = pd.DataFrame(report).transpose()
            report_path = 'classification_report_ensemble.csv'
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)
            
        return metrics

def run_ensemble_evaluation_refactored():
    """
    Orchestrates the complete Ensemble model evaluation process using the refactored class.
    This function will simulate data loading and cross-validation until dedicated modules are ready.
    """
    print("Running refactored Ensemble evaluation...")

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

    # Ensemble configuration
    ensemble_config = {
        'base_models_config': {
            'baseline': {
                'model_params': {
                    'solver': 'liblinear', 'C': 1.0, 'max_iter': 1000, 'multi_class': 'ovr', 'random_state': 42
                }
            },
            'xgboost': {
                'model_params': {
                    'objective': 'multi:softprob', 'num_class': len(np.unique(y_data)), 'learning_rate': 0.1,
                    'max_depth': 6, 'seed': 42
                },
                'num_boost_round': 100, 'early_stopping_rounds': 10
            },
            'lightgbm': {
                'model_params': {
                    'objective': 'multiclass', 'num_class': len(np.unique(y_data)), 'learning_rate': 0.1,
                    'max_depth': 8, 'seed': 42
                },
                'num_boost_round': 100, 'early_stopping_rounds': 10
            },
            'catboost': {
                'model_params': {
                    'loss_function': 'MultiClass', 'iterations': 100, 'depth': 6, 'learning_rate': 0.1,
                    'random_seed': 42, 'num_class': len(np.unique(y_data))
                }
            },
            'neural_network': {
                'model_params': {
                    'layer1_units': 256, 'dropout1_rate': 0.3, 'learning_rate': 0.001
                },
                'training_params': {
                    'epochs': 5, # Reduced epochs for faster simulation
                    'batch_size': 256, 'validation_split': 0.2, 'early_stopping_patience': 2
                }
            }
        },
        'voting_params': {
            'voting': 'soft',
            'n_jobs': -1
        }
    }

    # Initialize the refactored Ensemble model
    ensemble_model = EnsembleModel(config=ensemble_config)
    
    all_y_test = []
    all_y_pred = []
    
    # Start MLflow run
    with mlflow.start_run(run_name="Refactored_Ensemble_Model"):
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
            print(f"\n--- Fold {fold_idx+1}/{len(cv_folds)} ---")
            
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            # Train the ensemble for the current fold
            ensemble_model.train(X_train, y_train, fold_idx=fold_idx)
            
            # Make predictions
            y_pred = ensemble_model.predict(X_test)
            
            # Evaluate and log metrics for the fold
            metrics = ensemble_model.evaluate(y_test, y_pred, fold_idx=fold_idx)
            
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
        ensemble_model.evaluate(pd.Series(all_y_test), pd.Series(all_y_pred), log_artifacts=True)
        
        # Log the ensemble classifier (the last trained one from the CV loop)
        mlflow.sklearn.log_model(ensemble_model.ensemble_classifier, "refactored_ensemble_model")
        
        print("\nRefactored Ensemble model evaluation completed and logged to MLflow!")

if __name__ == "__main__":
    run_ensemble_evaluation_refactored()
