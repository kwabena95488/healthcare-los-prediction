from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, List, Tuple, Union

class BaseModel(ABC):
    """
    Abstract base class for all models in the healthcare analytics project.

    This class defines a common interface for training, prediction, and evaluation
    to ensure consistency across different model implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the base model with configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing model configuration parameters.
        """
        self.config = config

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Trains the model using the provided training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            **kwargs: Additional keyword arguments for training.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs: Any) -> Union[pd.Series, pd.DataFrame]:
        """
        Makes predictions on the input data.

        Args:
            X (pd.DataFrame): Input features for prediction.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
            Union[pd.Series, pd.DataFrame]: Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: Union[pd.Series, pd.DataFrame], **kwargs: Any) -> Dict[str, float]:
        """
        Evaluates the model's predictions against the true labels.

        Args:
            y_true (pd.Series): True labels.
            y_pred (Union[pd.Series, pd.DataFrame]): Predicted values.
            **kwargs: Additional keyword arguments for evaluation.

        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        pass

    # Optional: Add methods for saving/loading models if needed across all models
    # @abstractmethod
    # def save(self, path: str) -> None:
    #     """
    #     Saves the trained model to a specified path.
    #     """
    #     pass

    # @abstractmethod
    # def load(self, path: str) -> None:
    #     """
    #     Loads a trained model from a specified path.
    #     """
    #     pass
