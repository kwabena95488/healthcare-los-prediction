"""
Configuration management utilities for the healthcare analytics project.

This module provides utilities for loading and managing configuration files,
environment variables, and application settings.

@module config
@version 1.0.0
@public
"""

import os
import yaml
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional

def get_project_root() -> Path:
    """
    Get the project root directory path.
    
    @returns {Path} Path to the project root directory
    @version 1.0.0
    @performance O(1) - Constant time complexity
    @public
    
    @example
    # Get project root
    root_path = get_project_root()
    """
    return Path(__file__).parent.parent.parent

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    @param {string} config_path - Path to the YAML configuration file
    @returns {Dict[str, Any]} Configuration dictionary
    @throws {FileNotFoundError} When configuration file is not found
    @throws {yaml.YAMLError} When YAML parsing fails
    @version 1.0.0
    @public
    
    @example
    # Load main configuration
    config = load_yaml_config('config/config.yaml')
    """
    project_root = get_project_root()
    full_path = project_root / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    
    with open(full_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def load_config() -> Dict[str, Any]:
    """
    Load the main application configuration.
    
    @returns {Dict[str, Any]} Main configuration dictionary
    @version 1.0.0
    @public
    
    @example
    # Load main configuration
    config = load_config()
    data_path = config['data']['raw_data_path']
    """
    return load_yaml_config('config/config.yaml')

def load_model_config() -> Dict[str, Any]:
    """
    Load model-specific configuration.
    
    @returns {Dict[str, Any]} Model configuration dictionary
    @version 1.0.0
    @public
    
    @example
    # Load model configuration
    model_config = load_model_config()
    xgb_params = model_config['xgboost']
    """
    return load_yaml_config('config/model_config.yaml')

def setup_logging() -> None:
    """
    Set up logging configuration from YAML file.
    
    @version 1.0.0
    @public
    
    @example
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    """
    project_root = get_project_root()
    
    # Create logs directory if it doesn't exist
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Load logging configuration
    logging_config = load_yaml_config('config/logging.yaml')
    logging.config.dictConfig(logging_config)

def get_data_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Get the data directory path from configuration.
    
    @param {Dict[str, Any]} [config] - Configuration dictionary (optional)
    @returns {Path} Path to the data directory
    @version 1.0.0
    @public
    
    @example
    # Get data path
    data_path = get_data_path()
    raw_data_path = data_path / 'raw'
    """
    if config is None:
        config = load_config()
    
    project_root = get_project_root()
    return project_root / config['data']['raw_data_path']

def get_model_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Get the models directory path from configuration.
    
    @param {Dict[str, Any]} [config] - Configuration dictionary (optional)
    @returns {Path} Path to the models directory
    @version 1.0.0
    @public
    
    @example
    # Get model path
    model_path = get_model_path()
    xgb_model_path = model_path / 'xgboost'
    """
    if config is None:
        config = load_config()
    
    project_root = get_project_root()
    return project_root / config['paths']['models']

def get_results_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Get the results directory path from configuration.
    
    @param {Dict[str, Any]} [config] - Configuration dictionary (optional)
    @returns {Path} Path to the results directory
    @version 1.0.0
    @public
    
    @example
    # Get results path
    results_path = get_results_path()
    figures_path = results_path / 'figures'
    """
    if config is None:
        config = load_config()
    
    project_root = get_project_root()
    return project_root / config['paths']['results'] 