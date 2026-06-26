"""
Self-contained training + asset-build script for the Healthcare LOS demo.

Produces a clean, reproducible LightGBM model and all artifacts the README,
the Gradio app, and the docs/img plots depend on. Preprocessing is fully
self-contained here (no dependency on prior pipeline-version params) so the
saved model and the inference path in app.py are guaranteed to agree.

Outputs:
  models/demo/lightgbm_demo.pkl        trained classifier
  models/demo/preprocessor.pkl         label encoders + feature order + metadata
  models/demo/metrics.json             verified holdout metrics
  docs/img/confusion_matrix.png        holdout confusion matrix
  docs/img/feature_importance.png      LightGBM gain importance
  docs/img/shap_summary.png            SHAP beeswarm on a sample
  docs/img/model_comparison.png        verified per-model comparison bar chart

Usage:
  python scripts/build_demo.py --data data/raw/train.csv
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Raw columns -> roles
ID_COLS = ["case_id", "patientid"]
CATEGORICAL = ["Hospital_type_code", "Hospital_region_code", "Department",
               "Ward_Type", "Ward_Facility_Code", "Type of Admission",
               "Severity of Illness"]
AGE_MAP = {"0-10": 0, "11-20": 1, "21-30": 2, "31-40": 3, "41-50": 4,
           "51-60": 5, "61-70": 6, "71-80": 7, "81-90": 8, "91-100": 9,
           "More than 100 Days": 10}
STAY_CLASSES = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60",
                "61-70", "71-80", "81-90", "91-100", "More than 100 Days"]
TARGET = "Stay"


def preprocess(df: pd.DataFrame, encoders: dict | None, fit: bool):
    """Clean + encode. Returns (X, encoders). Deterministic and self-contained."""
    df = df.copy()
    for c in ID_COLS:
        df.drop(columns=c, inplace=True, errors="ignore")

    # Impute the two columns with known missingness using the training mode.
    if fit:
        encoders = {"modes": {}, "cat_maps": {}, "feature_order": None}
        for c in ["Bed Grade", "City_Code_Patient"]:
            encoders["modes"][c] = float(df[c].mode().iloc[0])
    for c in ["Bed Grade", "City_Code_Patient"]:
        df[c] = df[c].fillna(encoders["modes"][c])

    # Ordinal age
    df["Age"] = df["Age"].map(AGE_MAP).astype("Int64")

    # Deterministic label-encoding for categoricals (unseen -> -1 at inference)
    for c in CATEGORICAL:
        if fit:
            cats = sorted(df[c].dropna().astype(str).unique())
            encoders["cat_maps"][c] = {v: i for i, v in enumerate(cats)}
        m = encoders["cat_maps"][c]
        df[c] = df[c].astype(str).map(m).fillna(-1).astype(int)

    if fit:
        encoders["feature_order"] = [c for c in df.columns if c != TARGET]
    X = df[encoders["feature_order"]].astype(float)
    return X, encoders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train.csv")
    ap.add_argument("--sample-rows", type=int, default=0,
                    help="If >0, train on a random sample (for quick smoke runs).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_model = root / "models" / "demo"
    out_img = root / "docs" / "img"
    out_model.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pd.read_csv(root / args.data)
    if args.sample_rows:
        df = df.sample(args.sample_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  rows={len(df):,}  cols={df.shape[1]}")

    y = df[TARGET].map({c: i for i, c in enumerate(STAY_CLASSES)})
    X_all, encoders = preprocess(df, None, fit=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    from lightgbm import LGBMClassifier
    print("Training LightGBM ...")
    model = LGBMClassifier(
        objective="multiclass", num_class=11, n_estimators=400,
        learning_rate=0.05, num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    metrics = {
        "model": "LightGBM (demo, self-contained)",
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1_weighted": float(f1_score(y_te, pred, average="weighted")),
        "f1_macro": float(f1_score(y_te, pred, average="macro")),
        "n_classes": 11,
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump(model, out_model / "lightgbm_demo.pkl")
    joblib.dump(encoders, out_model / "preprocessor.pkl")
    (out_model / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- Plots -------------------------------------------------------------
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(9, 8))
    cm = confusion_matrix(y_te, pred, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=STAY_CLASSES).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False, values_format=".2f")
    ax.set_title("LightGBM — Normalized Confusion Matrix (holdout)")
    plt.tight_layout(); plt.savefig(out_img / "confusion_matrix.png", dpi=120); plt.close()

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=encoders["feature_order"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.tail(15).plot.barh(ax=ax, color="#2b8cbe")
    ax.set_title("LightGBM — Top Feature Importances (gain)")
    plt.tight_layout(); plt.savefig(out_img / "feature_importance.png", dpi=120); plt.close()

    # SHAP beeswarm on a sample
    try:
        import shap
        samp = X_te.sample(min(2000, len(X_te)), random_state=RANDOM_STATE)
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(samp)
        plt.figure()
        shap.summary_plot(sv, samp, class_names=STAY_CLASSES, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(out_img / "shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()
        print("  SHAP summary saved.")
    except Exception as e:
        print(f"  SHAP plot skipped: {e}")

    # Verified model-comparison bar chart (saved-report numbers + this demo)
    comp = pd.DataFrame({
        "Model": ["Baseline\n(LogReg)", "XGBoost", "LightGBM", "Stacking\nEnsemble", "LightGBM\n(demo)"],
        "Accuracy": [0.376, 0.425, 0.426, 0.400, metrics["accuracy"]],
        "F1 weighted": [0.324, 0.391, 0.392, 0.383, metrics["f1_weighted"]],
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp)); w = 0.38
    ax.bar(x - w/2, comp["Accuracy"], w, label="Accuracy", color="#2b8cbe")
    ax.bar(x + w/2, comp["F1 weighted"], w, label="F1 (weighted)", color="#a6bddb")
    ax.set_xticks(x); ax.set_xticklabels(comp["Model"]); ax.set_ylim(0, 0.5)
    ax.set_title("Model Comparison — 11-class LOS (verified)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_img / "model_comparison.png", dpi=120); plt.close()

    print("Done. Artifacts in models/demo/ and docs/img/.")


if __name__ == "__main__":
    main()
