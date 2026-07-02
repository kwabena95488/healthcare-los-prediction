"""
Healthcare Length-of-Stay — live demo (Gradio).

Loads the self-contained LightGBM model trained by `scripts/build_demo.py`,
takes admission-time patient features, and returns:
  • the predicted Length-of-Stay bucket + class probabilities
  • a SHAP explanation of *that specific* prediction (top feature contributions)

Run locally:   python app.py
Deploy:        Hugging Face Spaces (SDK: gradio) — see README "Live demo".
"""
from __future__ import annotations

import warnings
from pathlib import Path

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "models" / "demo" / "lightgbm_demo.pkl")
PRE = joblib.load(ROOT / "models" / "demo" / "preprocessor.pkl")

# Built lazily on first prediction — keeps startup fast and avoids heavy work
# during the Space's health check.
_EXPLAINER = None


def _get_explainer():
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = shap.TreeExplainer(MODEL)
    return _EXPLAINER

STAY_CLASSES = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60",
                "61-70", "71-80", "81-90", "91-100", "More than 100 Days"]
AGE_MAP = {"0-10": 0, "11-20": 1, "21-30": 2, "31-40": 3, "41-50": 4,
           "51-60": 5, "61-70": 6, "71-80": 7, "81-90": 8, "91-100": 9,
           "More than 100 Days": 10}
CAT = PRE["cat_maps"]
ORDER = PRE["feature_order"]


def _featurize(raw: dict) -> pd.DataFrame:
    """Apply the exact training-time preprocessing to one record."""
    row = dict(raw)
    row["Age"] = AGE_MAP[row["Age"]]
    for c in ["Bed Grade", "City_Code_Patient"]:
        if row.get(c) is None:
            row[c] = PRE["modes"][c]
    for c, m in CAT.items():
        row[c] = m.get(str(row[c]), -1)
    return pd.DataFrame([[float(row[c]) for c in ORDER]], columns=ORDER)


def predict(hospital_code, hospital_type, city_hospital, region, extra_rooms,
            department, ward_type, ward_facility, bed_grade, city_patient,
            admission_type, severity, visitors, age, deposit):
    raw = {
        "Hospital_code": hospital_code, "Hospital_type_code": hospital_type,
        "City_Code_Hospital": city_hospital, "Hospital_region_code": region,
        "Available Extra Rooms in Hospital": extra_rooms, "Department": department,
        "Ward_Type": ward_type, "Ward_Facility_Code": ward_facility,
        "Bed Grade": bed_grade, "City_Code_Patient": city_patient,
        "Type of Admission": admission_type, "Severity of Illness": severity,
        "Visitors with Patient": visitors, "Age": age, "Admission_Deposit": deposit,
    }
    try:
        return _predict_impl(raw)
    except Exception as e:  # surface the error in the UI instead of a silent 503
        import traceback
        traceback.print_exc()
        return f"Error: {e}", {}, None


def _predict_impl(raw):
    X = _featurize(raw)
    proba = MODEL.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    label = f"{STAY_CLASSES[pred_idx]} days"
    probs = {STAY_CLASSES[i]: float(proba[i]) for i in range(len(STAY_CLASSES))}

    # SHAP explanation for the predicted class
    sv = _get_explainer().shap_values(X)
    # shap_values: list (per class) of (1, n_features) OR (1, n_features, n_classes)
    if isinstance(sv, list):
        contrib = np.asarray(sv[pred_idx])[0]
    else:
        contrib = np.asarray(sv)[0, :, pred_idx]
    s = pd.Series(contrib, index=ORDER).sort_values(key=np.abs).tail(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#d6604d" if v > 0 else "#4393c3" for v in s.values]
    s.plot.barh(ax=ax, color=colors)
    ax.set_title(f"Why '{STAY_CLASSES[pred_idx]}'? Top SHAP contributions")
    ax.set_xlabel("← shorter stay      SHAP value      longer stay →")
    plt.tight_layout()
    return label, probs, fig


with gr.Blocks(title="Hospital Length-of-Stay Predictor") as demo:
    gr.Markdown(
        "# 🏥 Hospital Length-of-Stay Predictor\n"
        "Predicts an admission's **Length of Stay** bucket (11 classes) from "
        "admission-time features, with a **SHAP explanation** of each prediction.\n\n"
        "_Model: LightGBM trained on the public Healthcare Analytics II dataset "
        "(synthetic, no real PII). Educational demo — not for clinical use._")
    with gr.Row():
        with gr.Column():
            hospital_code = gr.Slider(1, 32, value=19, step=1, label="Hospital code")
            hospital_type = gr.Dropdown(list(CAT["Hospital_type_code"]), value="a", label="Hospital type")
            city_hospital = gr.Slider(1, 13, value=5, step=1, label="City code (hospital)")
            region = gr.Dropdown(list(CAT["Hospital_region_code"]), value="X", label="Hospital region")
            extra_rooms = gr.Slider(0, 24, value=3, step=1, label="Available extra rooms")
            department = gr.Dropdown(list(CAT["Department"]), value="gynecology", label="Department")
            ward_type = gr.Dropdown(list(CAT["Ward_Type"]), value="R", label="Ward type")
            ward_facility = gr.Dropdown(list(CAT["Ward_Facility_Code"]), value="F", label="Ward facility")
        with gr.Column():
            bed_grade = gr.Slider(1, 4, value=2, step=1, label="Bed grade")
            city_patient = gr.Slider(1, 38, value=8, step=1, label="City code (patient)")
            admission_type = gr.Dropdown(list(CAT["Type of Admission"]), value="Trauma", label="Admission type")
            severity = gr.Dropdown(list(CAT["Severity of Illness"]), value="Moderate", label="Severity of illness")
            visitors = gr.Slider(0, 32, value=3, step=1, label="Visitors with patient")
            age = gr.Dropdown(list(AGE_MAP.keys()), value="41-50", label="Age band")
            deposit = gr.Slider(1800, 11000, value=4700, step=50, label="Admission deposit")
    btn = gr.Button("Predict length of stay", variant="primary")
    with gr.Row():
        out_label = gr.Label(label="Predicted length of stay")
        out_probs = gr.Label(num_top_classes=5, label="Class probabilities")
    out_plot = gr.Plot(label="SHAP explanation for this prediction")

    inputs = [hospital_code, hospital_type, city_hospital, region, extra_rooms,
              department, ward_type, ward_facility, bed_grade, city_patient,
              admission_type, severity, visitors, age, deposit]
    btn.click(predict, inputs=inputs, outputs=[out_label, out_probs, out_plot])

if __name__ == "__main__":
    # ssr_mode=False: Gradio 6's server-side rendering is enabled by default on
    # HF Spaces (Node present) and can break the queue SSE stream; disable it.
    demo.queue().launch(ssr_mode=False)
