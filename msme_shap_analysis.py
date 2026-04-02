from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import msme_training_pipeline as m


DATASET_PATH = "data/final_dataset.csv"
MODEL_BUNDLE_PATH = "msme_outputs/msme_model_bundle.joblib"
OUTPUT_DIR = "msme_outputs/shap_outputs"

# Number of rows to use for SHAP background and explanations.
# Increase these for more stable results, decrease if runtime is slow.
BACKGROUND_SIZE = 300
EXPLAIN_SIZE = 400

# Which explained row to create a local waterfall plot for.
ROW_TO_EXPLAIN = 0

# Extra features for dependence plots.
DEPENDENCE_FEATURES = ["Revenue_latest", "Expenditure_latest", "Staff_latest", "Profit_latest"]

def load_bundle(bundle_path: str | Path) -> dict:
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Saved model bundle not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    required = {"model", "raw_input_columns", "feature_columns", "model_name", "is_calibrated"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Bundle is missing keys: {sorted(missing)}")
    return bundle


def failed_probability_function(model, feature_columns: list[str]):
    def predict_failed_proba(x_like):
        if isinstance(x_like, pd.DataFrame):
            X_df = x_like.copy()
        else:
            X_df = pd.DataFrame(x_like, columns=feature_columns)

        # Keep the exact column order used during training.
        X_df = X_df[feature_columns]

        probs = model.predict_proba(X_df)
        class_to_col = {int(cls): i for i, cls in enumerate(model.classes_)}
        failed_col = class_to_col[1]
        return probs[:, failed_col]

    return predict_failed_proba


def save_matplotlib(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main(
    dataset_path: str = DATASET_PATH,
    model_bundle_path: str = MODEL_BUNDLE_PATH,
    output_dir: str = OUTPUT_DIR,
) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(model_bundle_path)
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    raw_input_columns = list(bundle["raw_input_columns"])

    df = m.load_dataset(dataset_path).copy()
    X = m.build_feature_frame(df[raw_input_columns]).copy()
    X = X[feature_columns]

    y_failed = (pd.to_numeric(df["Business_Status"], errors="coerce") == 0).astype(int)

    predict_failed_proba = failed_probability_function(model, feature_columns)

    background = X.sample(min(BACKGROUND_SIZE, len(X)), random_state=42)
    X_explain = X.sample(min(EXPLAIN_SIZE, len(X)), random_state=42).reset_index(drop=True)

    # This explains the exact saved model output, including calibration if the saved model is calibrated.
    explainer = shap.Explainer(
        predict_failed_proba,
        masker=background,
        algorithm="permutation",
        feature_names=feature_columns,
    )
    shap_values = explainer(X_explain)

    # Save SHAP values and feature values.
    shap_df = pd.DataFrame(shap_values.values, columns=feature_columns)
    shap_df.to_csv(outdir / "shap_values.csv", index=False)
    X_explain.to_csv(outdir / "shap_feature_sample.csv", index=False)

    # Global feature importance.
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            "mean_shap": shap_values.values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(outdir / "shap_feature_importance.csv", index=False)

    # Global bar plot.
    plt.figure(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=20, show=False)
    save_matplotlib(outdir / "shap_bar_importance.png")

    # Beeswarm plot.
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    save_matplotlib(outdir / "shap_beeswarm.png")

    # Summary dot plot alternative.
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values.values, X_explain, show=False, max_display=20)
    save_matplotlib(outdir / "shap_summary_dot.png")

    # Local waterfall plot for one selected row.
    row_id = min(max(ROW_TO_EXPLAIN, 0), len(X_explain) - 1)
    local_prob = float(predict_failed_proba(X_explain.iloc[[row_id]])[0])

    local_contrib_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_value": X_explain.iloc[row_id].values,
            "shap_value": shap_values.values[row_id],
            "abs_shap_value": np.abs(shap_values.values[row_id]),
        }
    ).sort_values("abs_shap_value", ascending=False)
    local_contrib_df.to_csv(outdir / "shap_local_contributions_row0.csv", index=False)

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[row_id], max_display=20, show=False)
    save_matplotlib(outdir / "shap_waterfall_row0.png")

    # Dependence plots for key features.
    for feature in DEPENDENCE_FEATURES:
        if feature not in X_explain.columns:
            continue
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            feature,
            shap_values.values,
            X_explain,
            interaction_index="auto",
            show=False,
        )
        save_matplotlib(outdir / f"dependence_{feature}.png")

    # A compact run summary.
    summary = {
        "dataset_path": str(dataset_path),
        "model_bundle_path": str(model_bundle_path),
        "model_name": bundle["model_name"],
        "use_calibrated": bool(bundle["is_calibrated"]),
        "rows_in_dataset": int(len(df)),
        "rows_explained": int(len(X_explain)),
        "background_rows": int(len(background)),
        "failure_rate_in_dataset": float(y_failed.mean()),
        "local_row_index_in_explain_sample": int(row_id),
        "local_predicted_failure_probability": local_prob,
        "top_10_features": importance_df.head(10).to_dict(orient="records"),
        "note": (
            "This script explains the exact saved model output. "
            "If the saved model is calibrated, the SHAP values reflect the calibrated failed probability."
        ),
    }

    with open(outdir / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved SHAP outputs to: {outdir}")
    print(f"Model used: {bundle['model_name']} | calibrated: {bundle['is_calibrated']}")


if __name__ == "__main__":
    main()
