#!/usr/bin/env python3
"""
Full Pipeline Script for Healthcare Length of Stay Prediction

This script orchestrates the complete machine learning pipeline including
data loading, preprocessing, training, evaluation, and reporting.

@script run_pipeline
@version 1.0.0
@public

Usage:
    python scripts/run_pipeline.py --config config/config.yaml
    python scripts/run_pipeline.py --models xgboost,lightgbm --skip-preprocessing
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import setup_logging, load_config, get_data_path, get_model_path, get_results_path
from src.data.make_dataset import load_raw_data, load_processed_data
from src.data.preprocessing import preprocess_features, split_data
from src.models.train_model import train_models, ModelTrainer
from src.evaluation.metrics import calculate_comprehensive_metrics, compare_models
from src.visualization.model_plots import generate_evaluation_plots

logger = logging.getLogger(__name__)

class MLPipeline:
    """
    Complete ML Pipeline for Healthcare Length of Stay Prediction
    
    @class MLPipeline
    @version 1.0.0
    @public
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ML Pipeline.
        
        @param {Dict[str, Any]} config - Configuration dictionary
        @version 1.0.0
        @public
        """
        self.config = config
        self.results = {}
        self.start_time = datetime.now()
        self.pipeline_id = f"pipeline_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize paths
        self.data_path = get_data_path(config)
        self.model_path = get_model_path(config)
        self.results_path = get_results_path(config)
        
        logger.info(f"Pipeline initialized: {self.pipeline_id}")
    
    def run_data_loading(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load and validate data.
        
        @param {bool} force_reload - Whether to force reload of processed data
        @returns {Dict[str, Any]} Data loading results
        @version 1.0.0
        @public
        """
        logger.info("=" * 50)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 50)
        
        try:
            step_start = time.time()
            
            # Check for processed data first
            processed_data_path = self.data_path / 'processed'
            
            if not force_reload and (processed_data_path / 'X_preprocessed.csv').exists():
                logger.info("Loading existing processed data...")
                X, y = load_processed_data(self.config)
                data_source = "processed"
            else:
                logger.info("Loading raw data...")
                train_data, test_data = load_raw_data(self.config)
                
                # Combine for preprocessing
                if test_data is not None:
                    combined_data = pd.concat([train_data, test_data], ignore_index=True)
                else:
                    combined_data = train_data
                
                X = combined_data.drop(columns=[self.config.get('target_column', 'lengthofstay')])
                y = combined_data[self.config.get('target_column', 'lengthofstay')]
                data_source = "raw"
            
            step_time = time.time() - step_start
            
            results = {
                'status': 'success',
                'data_source': data_source,
                'shape': X.shape,
                'target_distribution': y.value_counts().to_dict(),
                'missing_values': X.isnull().sum().to_dict(),
                'execution_time': step_time
            }
            
            logger.info(f"Data loaded successfully: {X.shape}")
            logger.info(f"Target distribution: {results['target_distribution']}")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            self.results['data_loading'] = results
            return X, y, results
            
        except Exception as e:
            error_msg = f"Data loading failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.results['data_loading'] = {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - step_start
            }
            raise RuntimeError(error_msg) from e
    
    def run_preprocessing(self, X: pd.DataFrame, y: pd.Series, 
                         skip_preprocessing: bool = False) -> Dict[str, Any]:
        """
        Preprocess features and split data.
        
        @param {pd.DataFrame} X - Input features
        @param {pd.Series} y - Target variable
        @param {bool} skip_preprocessing - Whether to skip preprocessing
        @returns {Dict[str, Any]} Preprocessing results
        @version 1.0.0
        @public
        """
        logger.info("=" * 50)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 50)
        
        try:
            step_start = time.time()
            
            if skip_preprocessing:
                logger.info("Skipping preprocessing (using data as-is)")
                X_processed = X
                preprocessing_steps = ["skipped"]
            else:
                logger.info("Applying preprocessing pipeline...")
                X_processed, preprocessing_steps = preprocess_features(X, self.config)
            
            # Split data
            logger.info("Splitting data into train/validation/test sets...")
            data_splits = split_data(X_processed, y, self.config)
            
            step_time = time.time() - step_start
            
            results = {
                'status': 'success',
                'preprocessing_steps': preprocessing_steps,
                'original_shape': X.shape,
                'processed_shape': X_processed.shape,
                'train_size': data_splits['X_train'].shape[0],
                'val_size': data_splits['X_val'].shape[0] if 'X_val' in data_splits else 0,
                'test_size': data_splits['X_test'].shape[0],
                'execution_time': step_time
            }
            
            logger.info(f"Preprocessing completed: {X.shape} -> {X_processed.shape}")
            logger.info(f"Data splits - Train: {results['train_size']}, Val: {results['val_size']}, Test: {results['test_size']}")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            self.results['preprocessing'] = results
            return data_splits, results
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.results['preprocessing'] = {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - step_start
            }
            raise RuntimeError(error_msg) from e
    
    def run_training(self, data_splits: Dict[str, Any], 
                    selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train machine learning models.
        
        @param {Dict[str, Any]} data_splits - Training/validation/test data splits
        @param {Optional[List[str]]} selected_models - List of models to train
        @returns {Dict[str, Any]} Training results
        @version 1.0.0
        @public
        """
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 50)
        
        try:
            step_start = time.time()
            
            # Initialize trainer
            trainer = ModelTrainer(self.config)
            
            # Train models
            if selected_models:
                logger.info(f"Training selected models: {selected_models}")
                training_results = trainer.train_selected_models(
                    data_splits, selected_models
                )
            else:
                logger.info("Training all configured models...")
                training_results = trainer.train_all_models(data_splits)
            
            step_time = time.time() - step_start
            
            # Extract model performance summary
            model_summary = {}
            for model_name, result in training_results.items():
                if result['status'] == 'success':
                    model_summary[model_name] = {
                        'training_time': result.get('training_time', 0),
                        'validation_score': result.get('validation_score', 0),
                        'model_path': result.get('model_path', '')
                    }
            
            results = {
                'status': 'success',
                'models_trained': len([r for r in training_results.values() if r['status'] == 'success']),
                'models_failed': len([r for r in training_results.values() if r['status'] == 'failed']),
                'model_summary': model_summary,
                'detailed_results': training_results,
                'execution_time': step_time
            }
            
            logger.info(f"Training completed: {results['models_trained']} successful, {results['models_failed']} failed")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            self.results['training'] = results
            return training_results, results
            
        except Exception as e:
            error_msg = f"Training failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.results['training'] = {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - step_start
            }
            raise RuntimeError(error_msg) from e
    
    def run_evaluation(self, training_results: Dict[str, Any], 
                      data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate trained models.
        
        @param {Dict[str, Any]} training_results - Results from training step
        @param {Dict[str, Any]} data_splits - Data splits for evaluation
        @returns {Dict[str, Any]} Evaluation results
        @version 1.0.0
        @public
        """
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 50)
        
        try:
            step_start = time.time()
            
            # Load successful models
            successful_models = {
                name: result for name, result in training_results.items()
                if result['status'] == 'success'
            }
            
            if not successful_models:
                raise ValueError("No successful models to evaluate")
            
            # Evaluate each model
            evaluation_results = {}
            
            for model_name, training_result in successful_models.items():
                logger.info(f"Evaluating {model_name}...")
                
                try:
                    # Load model
                    model_path = Path(training_result['model_path'])
                    model = self._load_model(model_path)
                    
                    # Calculate metrics
                    metrics = calculate_comprehensive_metrics(
                        model, 
                        data_splits['X_test'], 
                        data_splits['y_test'],
                        data_splits.get('X_val'),
                        data_splits.get('y_val')
                    )
                    
                    evaluation_results[model_name] = {
                        'status': 'success',
                        'metrics': metrics,
                        'model_path': str(model_path)
                    }
                    
                except Exception as e:
                    logger.error(f"Evaluation failed for {model_name}: {e}")
                    evaluation_results[model_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Compare models
            comparison_results = compare_models(evaluation_results)
            
            step_time = time.time() - step_start
            
            results = {
                'status': 'success',
                'models_evaluated': len([r for r in evaluation_results.values() if r['status'] == 'success']),
                'best_model': comparison_results.get('best_model'),
                'best_score': comparison_results.get('best_score'),
                'model_rankings': comparison_results.get('rankings', []),
                'detailed_results': evaluation_results,
                'comparison': comparison_results,
                'execution_time': step_time
            }
            
            logger.info(f"Evaluation completed: {results['models_evaluated']} models evaluated")
            logger.info(f"Best model: {results['best_model']} (score: {results['best_score']:.3f})")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            self.results['evaluation'] = results
            return evaluation_results, results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.results['evaluation'] = {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - step_start
            }
            raise RuntimeError(error_msg) from e
    
    def run_reporting(self, evaluation_results: Dict[str, Any], 
                     data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate reports and visualizations.
        
        @param {Dict[str, Any]} evaluation_results - Results from evaluation step
        @param {Dict[str, Any]} data_splits - Data splits for visualization
        @returns {Dict[str, Any]} Reporting results
        @version 1.0.0
        @public
        """
        logger.info("=" * 50)
        logger.info("STEP 5: REPORTING & VISUALIZATION")
        logger.info("=" * 50)
        
        try:
            step_start = time.time()
            
            # Generate evaluation plots
            plot_results = generate_evaluation_plots(
                evaluation_results, 
                data_splits, 
                self.results_path / 'figures'
            )
            
            # Save pipeline results
            pipeline_report_path = self.results_path / 'reports' / f'{self.pipeline_id}_report.json'
            pipeline_report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(pipeline_report_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Generate summary report
            summary_report = self._generate_summary_report()
            summary_path = self.results_path / 'reports' / f'{self.pipeline_id}_summary.txt'
            
            with open(summary_path, 'w') as f:
                f.write(summary_report)
            
            step_time = time.time() - step_start
            
            results = {
                'status': 'success',
                'plots_generated': len(plot_results.get('generated_plots', [])),
                'pipeline_report_path': str(pipeline_report_path),
                'summary_report_path': str(summary_path),
                'execution_time': step_time
            }
            
            logger.info(f"Reporting completed: {results['plots_generated']} plots generated")
            logger.info(f"Pipeline report saved to: {pipeline_report_path}")
            logger.info(f"Summary report saved to: {summary_path}")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            self.results['reporting'] = results
            return results
            
        except Exception as e:
            error_msg = f"Reporting failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.results['reporting'] = {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - step_start
            }
            raise RuntimeError(error_msg) from e
    
    def _load_model(self, model_path: Path) -> Any:
        """Load a trained model from file."""
        import joblib
        return joblib.load(model_path)
    
    def _generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        report = f"""
HEALTHCARE LENGTH OF STAY PREDICTION - PIPELINE REPORT
=====================================================

Pipeline ID: {self.pipeline_id}
Start Time: {self.start_time}
End Time: {end_time}
Total Execution Time: {total_time:.2f} seconds

PIPELINE STEPS SUMMARY
=====================
"""
        
        for step_name, step_results in self.results.items():
            status = step_results.get('status', 'unknown')
            exec_time = step_results.get('execution_time', 0)
            
            report += f"\n{step_name.upper()}: {status.upper()} ({exec_time:.2f}s)"
            
            if step_name == 'data_loading':
                report += f"\n  - Data shape: {step_results.get('shape')}"
                report += f"\n  - Data source: {step_results.get('data_source')}"
                
            elif step_name == 'training':
                report += f"\n  - Models trained: {step_results.get('models_trained', 0)}"
                report += f"\n  - Models failed: {step_results.get('models_failed', 0)}"
                
            elif step_name == 'evaluation':
                report += f"\n  - Models evaluated: {step_results.get('models_evaluated', 0)}"
                report += f"\n  - Best model: {step_results.get('best_model', 'N/A')}"
                report += f"\n  - Best score: {step_results.get('best_score', 0):.3f}"
        
        if 'evaluation' in self.results and 'model_rankings' in self.results['evaluation']:
            report += "\n\nMODEL PERFORMANCE RANKING\n========================\n"
            for i, model_info in enumerate(self.results['evaluation']['model_rankings'][:5], 1):
                report += f"{i}. {model_info['model']} - Score: {model_info['score']:.3f}\n"
        
        report += f"\n\nPipeline completed {'successfully' if all(r.get('status') == 'success' for r in self.results.values()) else 'with errors'}"
        
        return report

def main():
    """
    Main pipeline function with CLI interface.
    
    @version 1.0.0
    @public
    """
    parser = argparse.ArgumentParser(
        description="Run complete ML pipeline for healthcare length of stay prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default configuration
  python scripts/run_pipeline.py
  
  # Run with specific models
  python scripts/run_pipeline.py --models xgboost,lightgbm,ensemble
  
  # Skip preprocessing step
  python scripts/run_pipeline.py --skip-preprocessing
  
  # Force reload of data
  python scripts/run_pipeline.py --force-reload
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of models to train (e.g., xgboost,lightgbm)'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step (use existing processed data)'
    )
    
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force reload of raw data even if processed data exists'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step'
    )
    
    parser.add_argument(
        '--skip-reporting',
        action='store_true',
        help='Skip reporting and visualization step'
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
        config = load_config(args.config)
        
        # Initialize pipeline
        pipeline = MLPipeline(config)
        
        logger.info("Starting Healthcare Length of Stay Prediction Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Data Loading
        X, y, data_results = pipeline.run_data_loading(args.force_reload)
        
        # Step 2: Preprocessing
        data_splits, prep_results = pipeline.run_preprocessing(X, y, args.skip_preprocessing)
        
        # Step 3: Training
        selected_models = args.models.split(',') if args.models else None
        training_results, train_results = pipeline.run_training(data_splits, selected_models)
        
        # Step 4: Evaluation
        if not args.skip_evaluation:
            evaluation_results, eval_results = pipeline.run_evaluation(training_results, data_splits)
        else:
            logger.info("Skipping evaluation step")
            evaluation_results = {}
        
        # Step 5: Reporting
        if not args.skip_reporting:
            report_results = pipeline.run_reporting(evaluation_results, data_splits)
        else:
            logger.info("Skipping reporting step")
        
        # Final summary
        total_time = (datetime.now() - pipeline.start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Pipeline ID: {pipeline.pipeline_id}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Status: {'SUCCESS' if all(r.get('status') == 'success' for r in pipeline.results.values()) else 'PARTIAL SUCCESS'}")
        
        if 'evaluation' in pipeline.results and pipeline.results['evaluation'].get('best_model'):
            print(f"Best model: {pipeline.results['evaluation']['best_model']}")
            print(f"Best score: {pipeline.results['evaluation']['best_score']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 