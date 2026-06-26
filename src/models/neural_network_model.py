import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.tensorflow
import time
from typing import Any, Dict, List, Tuple, Union

from src.models.base_model import BaseModel
from src.data.preprocessing import apply_age_ordinal_mapping, get_feature_types, get_preprocessor, inverse_transform_predictions
# from src.data.make_dataset import load_data # Placeholder, will be implemented in T004
# from src.evaluation.cross_validation import create_stratified_kfold # Placeholder, will be implemented in T006

# Set up MLflow
experiment_name = "Healthcare_LOS_Prediction"
mlflow.set_experiment(experiment_name)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NeuralNetworkModel(BaseModel):
    """
    A Neural Network model for Healthcare Length of Stay Prediction.

    This model inherits from BaseModel and implements its abstract methods for
    training, prediction, and evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the NeuralNetworkModel with its architecture and training parameters.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
                                      Expected keys: 'model_params' for NN architecture,
                                      'training_params' for epochs, batch_size, callbacks,
                                      'label_encoder_classes' if pre-fitted.
        """
        super().__init__(config)
        self.model_params = self.config.get('model_params', {})
        self.training_params = self.config.get('training_params', {})
        
        self.label_encoder = LabelEncoder()
        if 'label_encoder_classes' in self.config:
            self.label_encoder.classes_ = np.array(self.config['label_encoder_classes'])

        self.model = None # Keras model
        self.preprocessor = None # Will be fitted during training
        self.preprocessor_config = self.config.get('preprocessing_params', {})

    def _create_model_architecture(self, input_dim: int, num_classes: int) -> tf.keras.Model:
        """
        Creates the neural network architecture based on model_params.
        """
        model = Sequential([
            Dense(self.model_params.get('layer1_units', 256), input_dim=input_dim, activation='relu'),
            BatchNormalization(),
            Dropout(self.model_params.get('dropout1_rate', 0.3)),
            
            Dense(self.model_params.get('layer2_units', 128), activation='relu'),
            BatchNormalization(),
            Dropout(self.model_params.get('dropout2_rate', 0.3)),
            
            Dense(self.model_params.get('layer3_units', 64), activation='relu'),
            BatchNormalization(),
            Dropout(self.model_params.get('dropout3_rate', 0.2)),
            
            Dense(self.model_params.get('layer4_units', 32), activation='relu'),
            BatchNormalization(),
            Dropout(self.model_params.get('dropout4_rate', 0.2)),
            
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.model_params.get('learning_rate', 0.001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Trains the Neural Network model, including fitting the preprocessor and label encoder.

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
            print("Target Class Mapping:", self.label_encoder.classes_)

        # Get and fit the preprocessor
        self.preprocessor = get_preprocessor(numeric_features, categorical_features, self.preprocessor_config)
        X_train_transformed = self.preprocessor.fit_transform(X_train_processed_age)
        
        # Create and compile the model
        self.model = self._create_model_architecture(X_train_transformed.shape[1], len(self.label_encoder.classes_))
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_params.get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_params.get('reduce_lr_factor', 0.5),
                patience=self.training_params.get('reduce_lr_patience', 5),
                min_lr=self.training_params.get('min_lr', 0.00001),
                verbose=1
            )
        ]

        # Train the model
        history = self.model.fit(
            X_train_transformed, y_train_encoded,
            epochs=self.training_params.get('epochs', 50),
            batch_size=self.training_params.get('batch_size', 256),
            validation_split=self.training_params.get('validation_split', 0.2),
            callbacks=callbacks,
            verbose=0 # Set to 0 for silent training, 2 for per-epoch output
        )
        print("NeuralNetworkModel training complete.")

        # MLflow logging for parameters (only once per run, not per fold)
        if fold_idx is None or fold_idx == 0: # Log params only for the first fold or if not in CV
            mlflow.log_params({
                "model_type": "NeuralNetwork",
                **self.model_params,
                **self.training_params
            })
            print(f"Logged model parameters to MLflow: {self.model_params}, {self.training_params}")
        
        # Plot training history
        if fold_idx is not None:
            plt.figure(figsize=(12, 5))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title(f'Accuracy - Fold {fold_idx+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title(f'Loss - Fold {fold_idx+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            history_file = f'nn_history_fold_{fold_idx+1}.png'
            plt.savefig(history_file)
            mlflow.log_artifact(history_file)
            plt.close()

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """
        Makes predictions using the trained Neural Network model.

        Args:
            X (pd.DataFrame): Input features for prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.Series: Predicted labels.
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor has not been trained. Call train() first.")
        
        print("Making predictions with NeuralNetworkModel...")
        X_processed_age = apply_age_ordinal_mapping(X)
        X_transformed = self.preprocessor.transform(X_processed_age)

        y_pred_proba = self.model.predict(X_transformed)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        
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
            plt.title('Confusion Matrix - Neural Network')
            plt.tight_layout()
            cm_path = 'confusion_matrix_nn.png'
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close() # Close plot to free memory
            
            # Log classification report
            report = classification_report(y_true_encoded, y_pred_encoded, output_dict=True, target_names=self.label_encoder.classes_)
            report_df = pd.DataFrame(report).transpose()
            report_path = 'classification_report_nn.csv'
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)
            
        return metrics

def run_neural_network_evaluation_refactored():
    """
    Orchestrates the complete Neural Network model evaluation process using the refactored class.
    This function will simulate data loading and cross-validation until dedicated modules are ready.
    """
    print("Running refactored Neural Network evaluation...")

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
            'layer1_units': 256, 'dropout1_rate': 0.3,
            'layer2_units': 128, 'dropout2_rate': 0.3,
            'layer3_units': 64, 'dropout3_rate': 0.2,
            'layer4_units': 32, 'dropout4_rate': 0.2,
            'learning_rate': 0.001
        },
        'training_params': {
            'epochs': 50,
            'batch_size': 256,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'reduce_lr_factor': 0.5,
            'reduce_lr_patience': 5,
            'min_lr': 0.00001
        }
    }

    # Initialize the refactored Neural Network model
    nn_model = NeuralNetworkModel(config=model_config)
    
    all_y_test = []
    all_y_pred = []
    
    # Start MLflow run
    with mlflow.start_run(run_name="Refactored_Neural_Network_Model"):
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
            print(f"\nTraining on fold {fold_idx+1}/{len(cv_folds)}")
            
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            # Train the model for the current fold
            nn_model.train(X_train, y_train, fold_idx=fold_idx)
            
            # Make predictions
            y_pred = nn_model.predict(X_test)
            
            # Evaluate and log metrics for the fold
            metrics = nn_model.evaluate(y_test, y_pred, fold_idx=fold_idx)
            
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
        nn_model.evaluate(pd.Series(all_y_test), pd.Series(all_y_pred), log_artifacts=True)
        
        # Log the model (the last trained model from the CV loop)
        # Note: Keras models need to be saved/loaded carefully.
        # For MLflow, log the Keras model directly.
        mlflow.tensorflow.log_model(nn_model.model, "refactored_neural_network_model")
        
        print("\nRefactored Neural Network model evaluation completed and logged to MLflow!")

if __name__ == "__main__":
    run_neural_network_evaluation_refactored()
