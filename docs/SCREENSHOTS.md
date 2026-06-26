# Screenshots & image capture guide

All README plots are **already generated and committed** to `docs/img/`. This
file documents how each was produced and how to (re)capture the live-app shot.

## Auto-generated plots (already committed)

Regenerate any of these with one command:

```bash
python scripts/build_demo.py --data data/raw/train.csv
```

| File | What it shows | How it's made |
|------|---------------|---------------|
| `docs/img/model_comparison.png` | Accuracy & weighted-F1 across all models | bar chart in `build_demo.py` |
| `docs/img/confusion_matrix.png` | LightGBM normalized confusion matrix (holdout) | `build_demo.py` |
| `docs/img/feature_importance.png` | Top-15 LightGBM gain importances | `build_demo.py` |
| `docs/img/shap_summary.png` | Global SHAP beeswarm (2k-row sample) | `build_demo.py` (SHAP TreeExplainer) |
| `docs/img/app_demo.png` | Per-prediction SHAP explanation from the app | `app.predict(...)` figure |

## Live-app screenshot (capture against your deployed Space)

The most compelling thumbnail is the **running app**. Capture it once deployed:

1. **Open** the app:
   - Local: `python app.py` → http://127.0.0.1:7860
   - Or your Hugging Face Space URL.
2. Leave the default inputs (or set: Age `41-50`, Admission type `Trauma`,
   Severity `Moderate`).
3. Click **"Predict length of stay"**.
4. Wait for the prediction, the class-probability bars, and the SHAP plot to render.
5. **Capture** the region from the *"Predict length of stay"* button down through
   the SHAP explanation (prediction + probabilities + SHAP all visible).
6. **Save as** `docs/img/app_live.png` and, if you want it as the README hero,
   reference it at the top of `README.md`.

### Suggested README hero (optional)

```markdown
![Live demo](docs/img/app_live.png)
```

## Recommended capture settings

- Browser window ≥ 1400 px wide so both input columns and the result row fit.
- Light or dark theme both read fine; the committed shots use Gradio's default dark.
- Export PNG; keep each image under ~500 KB (the SHAP beeswarm is the largest).
