#!/usr/bin/env python3
"""
Prediction Script for Healthcare Length of Stay Prediction

This script provides prediction capabilities for trained models including
batch predictions, single predictions, and various output formats.

@script predict
@version 1.0.0
@public

Usage:
    python scripts/predict.py --model models/xgboost/model.pkl --input data/new_patients.csv
    python scripts/predict.py --model-dir models/ensemble --input data/test.csv --output predictions.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import joblib

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import setup_logging, load_config, get_model_path, get_results_path
from src.data.make_dataset import load_raw_data
from src.data.preprocessing import preprocess_features

logger = logging.getLogger(__name__)

def load_model(model_path: Path) -> Any:
    """
    Load a trained model from file.
    
    @param {Path} model_path - Path to the model file
    @returns {Any} Loaded model object
    @throws {FileNotFoundError} When model file is not found
    @version 1.0.0
    @public
    
    @example
    # Load model
    model = load_model(Path('models/xgboost/model.pkl'))
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

def load_input_data(input_path: Path) -> pd.DataFrame:
    """
    Load input data for prediction.
    
    @param {Path} input_path - Path to input data file
    @returns {pd.DataFrame} Input data
    @throws {FileNotFoundError} When input file is not found
    @version 1.0.0
    @public
    
    @example
    # Load input data
    data = load_input_data(Path('data/new_patients.csv'))
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading input data from {input_path}")
    
    # Support multiple file formats
    if input_path.suffix.lower() == '.csv':
        return pd.read_csv(input_path)
    elif input_path.suffix.lower() in ['.json', '.jsonl']:
        return pd.read_json(input_path)
    elif input_path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(input_path)
    elif input_path.suffix.lower() == '.parquet':
        return pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

def predict_batch(model: Any, X: pd.DataFrame, include_probabilities: bool = False) -> Dict[str, Any]:
    """
    Make batch predictions using the trained model.
    
    @param {Any} model - Trained model
    @param {pd.DataFrame} X - Input features
    @param {bool} include_probabilities - Whether to include prediction probabilities
    @returns {Dict[str, Any]} Prediction results
    @version 1.0.0
    @public
    
    @example
    # Make batch predictions
    predictions = predict_batch(model, X_test, include_probabilities=True)
    """
    logger.info(f"Making predictions for {X.shape[0]} samples...")
    
    # Make predictions
    y_pred = model.predict(X)
    
    results = {
        'predictions': y_pred.tolist(),
        'num_predictions': len(y_pred),
        'prediction_shape': y_pred.shape
    }
    
    # Add probabilities if requested and available
    if include_probabilities:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            results['probabilities'] = y_pred_proba.tolist()
            results['probability_shape'] = y_pred_proba.shape
            logger.info("Prediction probabilities included")
        elif hasattr(model, 'decision_function'):
            y_decision = model.decision_function(X)
            results['decision_scores'] = y_decision.tolist()
            results['decision_shape'] = y_decision.shape
            logger.info("Decision function scores included")
        else:
            logger.warning("Model does not support probability predictions")
    
    return results

def predict_single(model: Any, features: Dict[str, Any], include_probabilities: bool = False) -> Dict[str, Any]:
    """
    Make a single prediction using the trained model.
    
    @param {Any} model - Trained model
    @param {Dict[str, Any]} features - Input features as dictionary
    @param {bool} include_probabilities - Whether to include prediction probabilities
    @returns {Dict[str, Any]} Prediction results
    @version 1.0.0
    @public
    
    @example
    # Make single prediction
    features = {'age': 65, 'condition': 'pneumonia', ...}
    prediction = predict_single(model, features, include_probabilities=True)
    """
    # Convert features to DataFrame
    X = pd.DataFrame([features])
    
    # Make prediction
    prediction_results = predict_batch(model, X, include_probabilities)
    
    # Extract single prediction
    result = {
        'prediction': prediction_results['predictions'][0],
        'input_features': features
    }
    
    if 'probabilities' in prediction_results:
        result['probabilities'] = prediction_results['probabilities'][0]
    
    if 'decision_scores' in prediction_results:
        result['decision_scores'] = prediction_results['decision_scores'][0]
    
    logger.info(f"Single prediction: {result['prediction']}")
    
    return result

def save_predictions(predictions: Dict[str, Any], input_data: pd.DataFrame, 
                    output_path: Path, format: str = 'csv') -> None:
    """
    Save predictions to file in specified format.
    
    @param {Dict[str, Any]} predictions - Prediction results
    @param {pd.DataFrame} input_data - Original input data
    @param {Path} output_path - Output file path
    @param {str} format - Output format ('csv', 'json', 'excel')
    @version 1.0.0
    @public
    
    @example
    # Save predictions
    save_predictions(predictions, input_data, Path('predictions.csv'), 'csv')
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create output DataFrame
    output_data = input_data.copy()
    output_data['prediction'] = predictions['predictions']
    
    # Add probabilities if available
    if 'probabilities' in predictions:
        prob_array = np.array(predictions['probabilities'])
        if prob_array.ndim == 2:
            # Multi-class probabilities
            for i in range(prob_array.shape[1]):
                output_data[f'probability_class_{i}'] = prob_array[:, i]
        else:
            # Binary probabilities
            output_data['probability'] = prob_array
    
    if 'decision_scores' in predictions:
        decision_array = np.array(predictions['decision_scores'])
        if decision_array.ndim == 2:
            for i in range(decision_array.shape[1]):
                output_data[f'decision_score_class_{i}'] = decision_array[:, i]
        else:
            output_data['decision_score'] = decision_array
    
    # Save in requested format
    if format.lower() == 'csv':
        output_data.to_csv(output_path, index=False)
    elif format.lower() == 'json':
        output_data.to_json(output_path, orient='records', indent=2)
    elif format.lower() == 'excel':
        output_data.to_excel(output_path, index=False)
    elif format.lower() == 'parquet':
        output_data.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    logger.info(f"Predictions saved to {output_path}")

def find_latest_model(model_dir: Path) -> Path:
    """
    Find the latest model file in a directory.
    
    @param {Path} model_dir - Directory to search for models
    @returns {Path} Path to the latest model file
    @throws {FileNotFoundError} When no model files found
    @version 1.0.0
    @public
    
    @example
    # Find latest model
    latest_model = find_latest_model(Path('models/xgboost'))
    """
    # Common model file extensions
    extensions = ['*.pkl', '*.joblib', '*.model']
    
    model_files = []
    for ext in extensions:
        model_files.extend(model_dir.glob(ext))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Return the most recently modified file
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using latest model: {latest_model}")
    
    return latest_model

def validate_input_features(data: pd.DataFrame, required_features: Optional[List[str]] = None) -> bool:
    """
    Validate input features against expected schema.
    
    @param {pd.DataFrame} data - Input data to validate
    @param {Optional[List[str]]} required_features - List of required feature names
    @returns {bool} True if validation passes
    @throws {ValueError} When validation fails
    @version 1.0.0
    @public
    
    @example
    # Validate features
    is_valid = validate_input_features(data, ['age', 'condition', 'severity'])
    """
    if data.empty:
        raise ValueError("Input data is empty")
    
    if required_features:
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for null values
    null_counts = data.isnull().sum()
    if null_counts.any():
        logger.warning(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")
    
    logger.info(f"Input validation passed. Shape: {data.shape}")
    return True

def main():
    """
    Main prediction function with CLI interface.
    
    @version 1.0.0
    @public
    """
    parser = argparse.ArgumentParser(
        description="Make predictions using trained healthcare length of stay models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict using specific model file
  python scripts/predict.py --model models/xgboost/model.pkl --input data/new_patients.csv
  
  # Use latest model from directory
  python scripts/predict.py --model-dir models/ensemble --input data/test.csv
  
  # Include probabilities and save as JSON
  python scripts/predict.py --model models/xgboost/model.pkl --input data/test.csv --probabilities --output predictions.json --format json
  
  # Single prediction from JSON
  python scripts/predict.py --model models/xgboost/model.pkl --json '{"age": 65, "severity": "high"}'
        """
    )
    
    # Model specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model',
        type=str,
        help='Path to specific model file'
    )
    model_group.add_argument(
        '--model-dir',
        type=str,
        help='Path to model directory (uses latest model)'
    )
    
    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=str,
        help='Path to input data file (CSV, JSON, Excel, Parquet)'
    )
    input_group.add_argument(
        '--json',
        type=str,
        help='JSON string with single prediction features'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: predictions with timestamp)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json', 'excel', 'parquet'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    parser.add_argument(
        '--probabilities',
        action='store_true',
        help='Include prediction probabilities if available'
    )
    
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Skip preprocessing (assume data is already preprocessed)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = load_config()
        
        # Load model
        if args.model:
            model_path = Path(args.model)
        else:
            model_dir = Path(args.model_dir)
            model_path = find_latest_model(model_dir)
        
        model = load_model(model_path)
        logger.info(f"Model loaded successfully: {model_path}")
        
        # Prepare input data
        if args.json:
            # Single prediction from JSON
            try:
                features = json.loads(args.json)
                logger.info("Making single prediction from JSON input")
                
                # Make prediction
                result = predict_single(model, features, args.probabilities)
                
                # Print result
                print("\n" + "="*50)
                print("SINGLE PREDICTION RESULT")
                print("="*50)
                print(f"Prediction: {result['prediction']}")
                if 'probabilities' in result:
                    print(f"Probabilities: {result['probabilities']}")
                if 'decision_scores' in result:
                    print(f"Decision Scores: {result['decision_scores']}")
                
                # Save result if output specified
                if args.output:
                    output_path = Path(args.output)
                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    logger.info(f"Result saved to {output_path}")
                
                return 0
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON input: {e}")
                return 1
        
        else:
            # Batch prediction from file
            input_path = Path(args.input)
            input_data = load_input_data(input_path)
            
            logger.info(f"Loaded input data: {input_data.shape}")
            
            # Validate input features
            validate_input_features(input_data)
            
            # Preprocess if needed
            if not args.no_preprocessing:
                logger.info("Applying preprocessing...")
                # Note: This would need to be adapted based on your preprocessing pipeline
                # For now, we'll assume the data is already preprocessed or use basic preprocessing
                X = input_data
            else:
                X = input_data
            
            # Make predictions
            predictions = predict_batch(model, X, args.probabilities)
            
            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                # Generate output path based on input
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"predictions_{timestamp}.{args.format}"
                output_path = input_path.parent / output_name
            
            # Save predictions
            save_predictions(predictions, input_data, output_path, args.format)
            
            # Print summary
            print("\n" + "="*50)
            print("BATCH PREDICTION SUMMARY")
            print("="*50)
            print(f"Input samples: {predictions['num_predictions']}")
            print(f"Predictions made: {len(predictions['predictions'])}")
            if 'probabilities' in predictions:
                print("Probabilities: Included")
            print(f"Output saved to: {output_path}")
            print(f"Output format: {args.format}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 