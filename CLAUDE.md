# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-class classification project predicting hospital patient **Length of Stay** (`Stay`, an 11-category bucket like `0-10`, `11-20`, ...) from admission-time features. It trains six model families (logistic-regression baseline, XGBoost, LightGBM, CatBoost, a TensorFlow MLP, and a stacking ensemble) and compares them. Best ensemble accuracy is ~43% — low absolute accuracy is inherent to the 11-class problem, not a bug; judge models on the relative table in `README.md` and `results/reports/`.

## Commands

```bash
make setup            # pip install -r requirements.txt + init logging
make install-dev      # editable install + pytest/black/isort/flake8/mypy/sphinx
make test             # pytest tests/ -v  (config in pytest.ini)
make lint             # flake8 src/ scripts/ tests/  +  mypy src/ scripts/
make format           # black + isort over src/ scripts/ tests/
make mlflow-ui        # browse experiments (experiments/mlruns)
make notebook         # jupyter lab notebooks/
```

Run a single test: `pytest tests/test_data_processing.py::TestClass::test_name -v`.
Tests run with `--cov-fail-under=80`, so the suite fails if coverage drops below 80%. Markers available: `unit`, `integration`, `slow`, `regression`, `smoke` (e.g. `pytest -m "not slow"`).

Training / inference (note the real script names below — they differ from `make`/README):

```bash
python scripts/train_model.py --model xgboost --config_path config/config.yaml
python scripts/run_pipeline.py --config config/config.yaml   # full end-to-end pipeline
python scripts/evaluate_model.py
python scripts/predict.py
```

`--model` choices: `baseline`, `xgboost`, `lightgbm`, `catboost`, `neural_network`, `ensemble`.

## Architecture

Two parallel surfaces describe the same work — keep them consistent when changing logic:

1. **`notebooks/01`–`05`** — the narrative, demo-facing pipeline (complete pipeline → EDA/preprocessing → training/eval → interpretation → prediction demo). These are the primary deliverable artifacts.
2. **`src/` + `scripts/`** — the refactored, packaged, testable version of that same pipeline.

Code flow in `src/`:
- `src/data/` — `make_dataset.py` (load raw/processed), `preprocessing.py`, `feature_engineering.py`, `validation.py`.
- `src/models/` — every model subclasses the ABC in `base_model.py` (`train`/`predict`/`evaluate`). Gradient-boosting models live under `src/models/gradient_boosting/`. `ensemble_model.py` stacks the base models.
- `src/evaluation/` — `metrics.py`, `cross_validation.py` (stratified k-fold), `model_comparison.py`, `interpretability.py` (SHAP / permutation importance).
- `src/visualization/visualize.py` — plotting.
- `src/utils/config.py` — central config loader; `load_config`, `setup_logging`, and path helpers (`get_data_path`, `get_model_path`, `get_results_path`). Most modules pull paths from `config/config.yaml` through here rather than hardcoding.

Config is YAML-driven: `config/config.yaml` (paths, data, MLflow), `config/model_config.yaml` (hyperparameters), `config/logging.yaml`. The target column (`Stay`), data paths, CV folds, and `random_state=42` all come from `config.yaml` — change behavior there, not in code.

Experiment tracking is MLflow, logging to `experiments/mlruns`. Trained artifacts go under `models/<name>/`; outputs (figures, metrics, reports, interpretability) under `results/`.

## Known drift — verify before relying on these

The docs/build files were written against a planned layout that the code partially diverged from. When something doesn't run, suspect this first:

- **`make predict` / `make reports` reference scripts that don't exist** (`scripts/make_predictions.py`, `scripts/generate_reports.py`). The real scripts are `scripts/predict.py` and `scripts/run_pipeline.py`.
- **`setup.py` console_scripts point at `scripts.make_predictions:main`** (missing) — the `healthcare-predict` entry point won't install cleanly.
- **`scripts/run_pipeline.py` imports `from src.models.train_model import train_models, ModelTrainer`**, but training logic lives in `scripts/train_model.py`; there is no `src/models/train_model.py`. The full pipeline script will not import as-is.
- README's "Project Structure" lists filenames (`baseline.py`, `neural_network.py`, `predict_model.py`, `eda_plots.py`, etc.) that don't match the actual files (`baseline_model.py`, `neural_network_model.py`, single `visualize.py`). Trust the filesystem over the README tree.

## Conventions

- Data files (`data/raw/train.csv`, `test.csv`) and `models/`, `results/`, `experiments/` artifacts are gitignored — only `.gitkeep`s and small markdown reports are tracked. Don't commit large data/model binaries.
- Column meanings: `data/raw/column-descriptions.md`. The target is `Stay`; `case_id`/`patientid` are identifiers, not features.
- Python 3.8+. Dependencies pinned loosely in `requirements.txt` (note `scipy==1.10.0` is the one hard pin).
