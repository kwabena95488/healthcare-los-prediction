#!/usr/bin/env python3
"""
Model Evaluation Script for Healthcare Length of Stay Prediction

This script provides comprehensive model evaluation capabilities including
performance metrics, model comparison, and report generation.

@script evaluate_model
@version 1.0.0
@public

Usage:
    python scripts/evaluate_model.py --model-dir models/xgboost
    python scripts/evaluate_model.py --compare-all --output results/evaluation_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import setup_logging, load_config, get_model_path, get_results_path
from src.data.make_dataset import load_processed_data
from src.evaluation.metrics import calculate_comprehensive_metrics
from src.visualization.model_plots import plot_confusion_matrix, plot_roc_curves

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

def evaluate_single_model(model_path: Path, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a single model and return comprehensive metrics.
    
    @param {Path} model_path - Path to the model file
    @param {pd.DataFrame} X_test - Test features
    @param {pd.Series} y_test - Test targets
    @returns {Dict[str, Any]} Evaluation results
    @version 1.0.0
    @public
    
    @example
    # Evaluate model
    results = evaluate_single_model(model_path, X_test, y_test)
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        
        # Calculate metrics
        results = {
            'model_path': str(model_path),
            'model_name': model_path.parent.name,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
            'f1_micro': float(f1_score(y_test, y_pred, average='micro')),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
        }
        
        # Add ROC AUC if probabilities available
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:  # Binary classification
                    if y_pred_proba.ndim == 2:
                        results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
                    else:
                        results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
                else:  # Multi-class
                    results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                results['roc_auc'] = None
        
        # Classification report
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        logger.info(f"Model {results['model_name']} - Accuracy: {results['accuracy']:.3f}, F1: {results['f1_weighted']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating model {model_path}: {e}")
        return {
            'model_path': str(model_path),
            'model_name': model_path.parent.name,
            'error': str(e)
        }

def find_model_files(models_dir: Path) -> List[Path]:
    """
    Find all model files in the models directory.
    
    @param {Path} models_dir - Path to models directory
    @returns {List[Path]} List of model file paths
    @version 1.0.0
    @public
    
    @example
    # Find model files
    model_files = find_model_files(Path('models'))
    """
    model_files = []
    
    # Common model file extensions
    extensions = ['*.pkl', '*.joblib', '*.model']
    
    for ext in extensions:
        model_files.extend(models_dir.glob(f"**/{ext}"))
    
    logger.info(f"Found {len(model_files)} model files")
    return model_files

def compare_models(model_files: List[Path], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Compare multiple models and return comparison results.
    
    @param {List[Path]} model_files - List of model file paths
    @param {pd.DataFrame} X_test - Test features
    @param {pd.Series} y_test - Test targets
    @returns {Dict[str, Any]} Comparison results
    @version 1.0.0
    @public
    
    @example
    # Compare models
    comparison = compare_models(model_files, X_test, y_test)
    """
    results = {}
    
    for model_file in model_files:
        logger.info(f"Evaluating {model_file}")
        model_results = evaluate_single_model(model_file, X_test, y_test)
        results[model_results['model_name']] = model_results
    
    # Create comparison summary
    comparison_df = pd.DataFrame([
        {
            'model': name,
            'accuracy': res.get('accuracy'),
            'f1_weighted': res.get('f1_weighted'),
            'f1_macro': res.get('f1_macro'),
            'roc_auc': res.get('roc_auc'),
        }
        for name, res in results.items()
        if 'error' not in res
    ])
    
    if not comparison_df.empty:
        # Rank models by F1 weighted score
        comparison_df = comparison_df.sort_values('f1_weighted', ascending=False)
        
        summary = {
            'best_model': comparison_df.iloc[0]['model'],
            'best_f1_weighted': comparison_df.iloc[0]['f1_weighted'],
            'model_ranking': comparison_df.to_dict('records'),
            'total_models_evaluated': len(comparison_df)
        }
    else:
        summary = {
            'error': 'No models could be evaluated successfully',
            'total_models_evaluated': 0
        }
    
    return {
        'summary': summary,
        'detailed_results': results,
        'comparison_table': comparison_df.to_dict('records') if not comparison_df.empty else []
    }

def save_evaluation_results(results: Dict[str, Any], output_path: Path) -> None:
    """
    Save evaluation results to file.
    
    @param {Dict[str, Any]} results - Evaluation results
    @param {Path} output_path - Output file path
    @version 1.0.0
    @public
    
    @example
    # Save results
    save_evaluation_results(results, Path('results/evaluation.json'))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {output_path}")

def main():
    """
    Main evaluation function with CLI interface.
    
    @version 1.0.0
    @public
    """
    parser = argparse.ArgumentParser(
        description="Evaluate healthcare length of stay prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a specific model
  python scripts/evaluate_model.py --model-dir models/xgboost
  
  # Compare all models
  python scripts/evaluate_model.py --compare-all
  
  # Evaluate with custom output
  python scripts/evaluate_model.py --compare-all --output results/my_evaluation.json
        """
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Path to specific model directory to evaluate'
    )
    
    parser.add_argument(
        '--model-file',
        type=str,
        help='Path to specific model file to evaluate'
    )
    
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare all models in the models directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results (default: results/metrics/evaluation_results.json)'
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
        
        # Load test data
        logger.info("Loading test data...")
        X, y = load_processed_data(config)
        
        # For demonstration, we'll use a train-test split since we don't have separate test data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Test set size: {X_test.shape}")
        
        # Determine what to evaluate
        if args.model_file:
            # Evaluate single model file
            model_path = Path(args.model_file)
            results = evaluate_single_model(model_path, X_test, y_test)
            
        elif args.model_dir:
            # Evaluate models in specific directory
            model_dir = Path(args.model_dir)
            model_files = find_model_files(model_dir)
            if not model_files:
                logger.error(f"No model files found in {model_dir}")
                return 1
            results = compare_models(model_files, X_test, y_test)
            
        elif args.compare_all:
            # Compare all models
            models_path = get_model_path(config)
            model_files = find_model_files(models_path)
            if not model_files:
                logger.error(f"No model files found in {models_path}")
                return 1
            results = compare_models(model_files, X_test, y_test)
            
        else:
            logger.error("Must specify --model-file, --model-dir, or --compare-all")
            parser.print_help()
            return 1
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            results_path = get_results_path(config)
            output_path = results_path / 'metrics' / 'evaluation_results.json'
        
        save_evaluation_results(results, output_path)
        
        # Print summary
        if 'summary' in results:
            summary = results['summary']
            print("\n" + "="*50)
            print("MODEL EVALUATION SUMMARY")
            print("="*50)
            print(f"Total models evaluated: {summary.get('total_models_evaluated', 0)}")
            if 'best_model' in summary:
                print(f"Best model: {summary['best_model']}")
                print(f"Best F1 Score: {summary['best_f1_weighted']:.3f}")
            print(f"Results saved to: {output_path}")
        else:
            print(f"\nEvaluation completed. Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 