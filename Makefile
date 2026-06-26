# Healthcare Length of Stay Prediction - Makefile
# Common commands for development, testing, and deployment

.PHONY: help install install-dev test train evaluate predict clean lint format docs setup

# Default target
help:
	@echo "Healthcare Length of Stay Prediction - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup           - Initial project setup (create directories, install deps)"
	@echo "  install         - Install the package in development mode"
	@echo "  install-dev     - Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test            - Run all tests"
	@echo "  test-data       - Run data processing tests"
	@echo "  test-models     - Run model tests"
	@echo "  lint            - Check code style and quality"
	@echo "  format          - Format code with black and isort"
	@echo ""
	@echo "Machine Learning Pipeline:"
	@echo "  train           - Train all models"
	@echo "  train-baseline  - Train baseline model only"
	@echo "  train-ensemble  - Train ensemble model"
	@echo "  evaluate        - Evaluate trained models"
	@echo "  predict         - Make predictions on test data"
	@echo ""
	@echo "Documentation and Reports:"
	@echo "  docs            - Generate documentation"
	@echo "  reports         - Generate analysis reports"
	@echo ""
	@echo "Utilities:"
	@echo "  clean           - Clean temporary files and caches"
	@echo "  clean-data      - Clean processed data files"
	@echo "  clean-models    - Clean trained model files"
	@echo ""

# Setup and Installation
setup:
	@echo "Setting up project environment..."
	pip install -r requirements.txt
	python -c "from src.utils.config import setup_logging; setup_logging()"
	@echo "Project setup complete!"

install:
	pip install -e .

install-dev:
	pip install -e .
	pip install pytest black isort flake8 mypy sphinx

# Testing
test:
	pytest tests/ -v

test-data:
	pytest tests/test_data/ -v

test-models:
	pytest tests/test_models/ -v

# Code Quality
lint:
	flake8 src/ scripts/ tests/
	mypy src/ scripts/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# Machine Learning Pipeline
train:
	python scripts/train_model.py --all

train-baseline:
	python scripts/train_model.py --model baseline

train-ensemble:
	python scripts/train_model.py --model ensemble

evaluate:
	python scripts/evaluate_model.py

predict:
	python scripts/make_predictions.py

# Data Processing
process-data:
	python -m src.data.make_dataset
	python -m src.data.preprocessing

# Documentation
docs:
	cd docs && make html

reports:
	python scripts/generate_reports.py

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

clean-data:
	rm -rf data/interim/*
	rm -rf data/processed/*

clean-models:
	rm -rf models/*/
	mkdir -p models/{baseline,xgboost,lightgbm,catboost,neural_network,ensemble,production}

# MLflow
mlflow-ui:
	mlflow ui --backend-store-uri experiments/mlruns

# Jupyter
notebook:
	jupyter lab notebooks/

# Docker (if using containerization)
docker-build:
	docker build -t healthcare-los .

docker-run:
	docker run -p 8888:8888 healthcare-los

# Package building
build:
	python setup.py sdist bdist_wheel

# Environment
env-export:
	pip freeze > requirements.txt

env-check:
	python -c "import sys; print(f'Python version: {sys.version}')"
	python -c "from src.utils.config import load_config; print('Config loading: OK')" 