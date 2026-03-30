
"""
MSME failure modelling pipeline

What this script fixes compared with the original notebook:
- removes leakage from full-data imputation
- avoids using Active_upto_Year as a predictor or masking signal
- keeps repeated IDs in the same split with grouped splitting
- uses a real train / validation / test workflow
- tunes models only on the training split
- calibrates the chosen model without touching the test split
- saves one deployable bundle for the dashboard

Target convention in this script:
    1 = Failed
    0 = Active
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
OBSERVATION_YEAR = 2024

RAW_REVENUE_COLS = [
    "Total_Revenue_four_years_before",
    "Total_Revenue_three_years_before",
    "Total_Revenue_two_years_before",
    "Total_Revenue_one_years_before",
]

RAW_EXPENDITURE_COLS = [
    "Total_expenditure_four_years_before",
    "Total_expenditure_three_years_before",
    "Total_expenditure_two_years_before",
    "Total_expenditure_one_years_before",
]

RAW_STAFF_COLS = [
    "Staff_four_years_before",
    "Staff_three_years_before",
    "Staff_two_years_before",
    "Staff_one_years_before",
]

RAW_INPUT_COLS = ["Open_Year"] + RAW_STAFF_COLS + RAW_EXPENDITURE_COLS + RAW_REVENUE_COLS
REQUIRED_DATASET_COLS = ["ID", "Business_Status"] + RAW_INPUT_COLS


def safe_divide_frame(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    denominator_values = denominator.to_numpy(dtype=float)
    denominator_values = np.where(denominator_values == 0, np.nan, denominator_values)
    divided = numerator.to_numpy(dtype=float) / denominator_values
    return pd.DataFrame(divided, index=numerator.index, columns=numerator.columns)


def last_observed_df(frame: pd.DataFrame) -> pd.Series:
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    for i, row in enumerate(values):
        valid = row[~np.isnan(row)]
        if valid.size:
            out[i] = valid[-1]
    return pd.Series(out, index=frame.index)


def trend_slope_df(frame: pd.DataFrame) -> pd.Series:
    x_full = np.arange(frame.shape[1], dtype=float).reshape(-1, 1)
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)

    for i, row in enumerate(values):
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        model = LinearRegression()
        model.fit(x_full[mask], row[mask])
        out[i] = float(model.coef_[0])

    return pd.Series(out, index=frame.index)


def build_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    revenue = df[RAW_REVENUE_COLS].apply(pd.to_numeric, errors="coerce")
    expenditure = df[RAW_EXPENDITURE_COLS].apply(pd.to_numeric, errors="coerce")
    staff = df[RAW_STAFF_COLS].apply(pd.to_numeric, errors="coerce")
    open_year = pd.to_numeric(df["Open_Year"], errors="coerce")

    profit = revenue.to_numpy(dtype=float) - expenditure.to_numpy(dtype=float)
    profit = pd.DataFrame(profit, columns=[f"Profit_{i}" for i in range(1, 5)], index=df.index)

    margin = safe_divide_frame(profit, revenue)
    margin.columns = [f"Margin_{i}" for i in range(1, 5)]

    features = pd.DataFrame(index=df.index)

    features["Revenue_latest"] = last_observed_df(revenue)
    features["Revenue_mean"] = revenue.mean(axis=1, skipna=True)
    features["Revenue_std"] = revenue.std(axis=1, skipna=True)

    features["Expenditure_latest"] = last_observed_df(expenditure)
    features["Expenditure_mean"] = expenditure.mean(axis=1, skipna=True)
    features["Expenditure_std"] = expenditure.std(axis=1, skipna=True)

    features["Staff_latest"] = last_observed_df(staff)
    features["Staff_mean"] = staff.mean(axis=1, skipna=True)
    features["Staff_std"] = staff.std(axis=1, skipna=True)

    features["Profit_latest"] = last_observed_df(profit)
    features["Profit_mean"] = profit.mean(axis=1, skipna=True)
    features["Profit_std"] = profit.std(axis=1, skipna=True)

    features["Profit_margin_latest"] = last_observed_df(margin)
    features["Profit_margin_mean"] = margin.mean(axis=1, skipna=True)

    features["Revenue_trend"] = trend_slope_df(revenue)
    features["Expenditure_trend"] = trend_slope_df(expenditure)
    features["Staff_trend"] = trend_slope_df(staff)
    features["Profit_trend"] = trend_slope_df(profit)

    features["Loss_years"] = (profit < 0).sum(axis=1)
    features["Business_Age"] = OBSERVATION_YEAR - open_year
    features["Open_Year_missing"] = open_year.isna().astype(int)

    features["Business_Age"] = features["Business_Age"].where(features["Business_Age"] >= 0, np.nan)

    return features


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing_cols = [c for c in REQUIRED_DATASET_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Remove exact duplicate records but keep repeated business IDs for grouped splitting.
    subset_cols = [c for c in df.columns]
    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)

    return df


def make_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_failed = (pd.to_numeric(df["Business_Status"], errors="coerce") == 0).astype(int)
    groups = df["ID"].astype(str)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(outer.split(df, y_failed, groups))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    y_train_val = (pd.to_numeric(train_val["Business_Status"], errors="coerce") == 0).astype(int)
    groups_train_val = train_val["ID"].astype(str)

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1)
    train_idx, val_idx = next(inner.split(train_val, y_train_val, groups_train_val))
    train_df = train_val.iloc[train_idx].reset_index(drop=True)
    val_df = train_val.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X_raw = df[RAW_INPUT_COLS].copy()
    X_feat = build_feature_frame(X_raw)
    y = (pd.to_numeric(df["Business_Status"], errors="coerce") == 0).astype(int)
    return X_feat, y


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob_failed: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_failed": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_failed": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_failed": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc_failed": float(roc_auc_score(y_true, y_prob_failed)),
        "pr_auc_failed": float(average_precision_score(y_true, y_prob_failed)),
        "brier_failed": float(brier_score_loss(y_true, y_prob_failed)),
    }


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], pd.DataFrame, str, List[List[int]]]:
    probs = model.predict_proba(X)
    class_to_col = {int(cls): i for i, cls in enumerate(model.classes_)}
    failed_col = class_to_col[1]
    y_prob_failed = probs[:, failed_col]
    y_pred = model.predict(X)

    metrics = compute_metrics(y, y_pred, y_prob_failed)
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(
        y,
        y_pred,
        labels=[0, 1],
        target_names=["Active", "Failed"],
        zero_division=0,
    )

    pred_frame = pd.DataFrame(
        {
            "true_label": y.values,
            "predicted_label": y_pred,
            "p_active": probs[:, class_to_col[0]],
            "p_failed": y_prob_failed,
        }
    )
    return metrics, pred_frame, report, cm


def make_model_search_spaces(scale_pos_weight: float) -> Dict[str, Tuple[Pipeline, Dict[str, List]]]:
    searches: Dict[str, Tuple[Pipeline, Dict[str, List]]] = {}

    logreg_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    logreg_grid = {
        "model__C": np.logspace(-3, 2, 12).tolist(),
    }
    searches["Logistic Regression"] = (logreg_pipe, logreg_grid)

    rf_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_grid = {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.8],
    }
    searches["Random Forest"] = (rf_pipe, rf_grid)

    xgb_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                xgb.XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    xgb_grid = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }
    searches["XGBoost"] = (xgb_pipe, xgb_grid)

    lgb_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                lgb.LGBMClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    verbose=-1,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    lgb_grid = {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63, 127],
        "model__max_depth": [-1, 5, 10, 20],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_samples": [10, 20, 40],
    }
    searches["LightGBM"] = (lgb_pipe, lgb_grid)

    return searches


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_iter: int,
) -> Dict[str, Dict]:
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = negative / positive if positive else 1.0

    searches = make_model_search_spaces(scale_pos_weight)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    groups = groups_train.astype(str).reset_index(drop=True)

    tuned = {}

    for model_name, (pipeline, param_grid) in searches.items():
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train, groups=groups)
        tuned[model_name] = {
            "search": search,
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
        }

    return tuned


def select_best_model(
    tuned_models: Dict[str, Dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict:
    comparison_rows = []

    for name, info in tuned_models.items():
        fitted = clone(info["best_estimator"])
        fitted.fit(X_train, y_train)

        val_metrics, _, _, _ = evaluate_model(fitted, X_val, y_val)
        comparison_rows.append(
            {
                "model_name": name,
                "cv_pr_auc_failed": info["best_cv_score"],
                **val_metrics,
            }
        )
        info["validation_metrics"] = val_metrics
        info["fitted_on_train"] = fitted

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["pr_auc_failed", "brier_failed", "roc_auc_failed"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    best_name = comparison_df.loc[0, "model_name"]
    best_info = tuned_models[best_name]
    best_info["validation_ranking"] = comparison_df
    return {"best_name": best_name, "best_info": best_info, "validation_table": comparison_df}


def calibrate_if_helpful(
    best_name: str,
    best_model_fitted,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict:
    base_metrics, _, base_report, base_cm = evaluate_model(best_model_fitted, X_val, y_val)

    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(best_model_fitted),
            method="sigmoid",
        )
    except Exception:
        calibrator = CalibratedClassifierCV(
            estimator=best_model_fitted,
            method="sigmoid",
            cv="prefit",
        )

    calibrator.fit(X_val, y_val)
    calibrated_metrics, _, calibrated_report, calibrated_cm = evaluate_model(calibrator, X_val, y_val)

    use_calibrated = calibrated_metrics["brier_failed"] < base_metrics["brier_failed"]

    return {
        "best_model_name": best_name,
        "uncalibrated_model": best_model_fitted,
        "calibrated_model": calibrator,
        "uncalibrated_validation_metrics": base_metrics,
        "calibrated_validation_metrics": calibrated_metrics,
        "uncalibrated_validation_report": base_report,
        "calibrated_validation_report": calibrated_report,
        "uncalibrated_validation_confusion_matrix": base_cm,
        "calibrated_validation_confusion_matrix": calibrated_cm,
        "use_calibrated": use_calibrated,
    }


def fit_final_model(
    best_estimator,
    use_calibrated: bool,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
):
    if use_calibrated:
        final_model = CalibratedClassifierCV(
            estimator=clone(best_estimator),
            method="sigmoid",
            cv=5,
        )
    else:
        final_model = clone(best_estimator)

    final_model.fit(X_train_val, y_train_val)
    return final_model


def main(
    csv_path: str = "final_dataset.csv",
    output_dir: str = "msme_outputs",
    n_iter: int = 20,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    train_df, val_df, test_df = make_splits(df)

    X_train, y_train = build_xy(train_df)
    X_val, y_val = build_xy(val_df)
    X_test, y_test = build_xy(test_df)

    tuned = tune_models(X_train, y_train, train_df["ID"], n_iter=n_iter)

    selection = select_best_model(tuned, X_train, y_train, X_val, y_val)
    best_name = selection["best_name"]
    best_info = selection["best_info"]

    calibration = calibrate_if_helpful(
        best_name=best_name,
        best_model_fitted=best_info["fitted_on_train"],
        X_val=X_val,
        y_val=y_val,
    )

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    final_model = fit_final_model(
        best_estimator=best_info["best_estimator"],
        use_calibrated=calibration["use_calibrated"],
        X_train_val=X_train_val,
        y_train_val=y_train_val,
    )

    test_metrics, test_predictions, test_report, test_cm = evaluate_model(final_model, X_test, y_test)

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "is_calibrated": calibration["use_calibrated"],
        "feature_columns": X_train.columns.tolist(),
        "raw_input_columns": RAW_INPUT_COLS,
        "target_definition": {"1": "Failed", "0": "Active"},
        "observation_year": OBSERVATION_YEAR,
    }

    joblib.dump(bundle, output / "msme_model_bundle.joblib")
    test_predictions.to_csv(output / "test_predictions.csv", index=False)

    summary = {
        "data": {
            "rows_after_exact_dedup": int(len(df)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "active_count_total": int((pd.to_numeric(df["Business_Status"], errors="coerce") == 1).sum()),
            "failed_count_total": int((pd.to_numeric(df["Business_Status"], errors="coerce") == 0).sum()),
        },
        "best_model": {
            "name": best_name,
            "best_cv_pr_auc_failed": best_info["best_cv_score"],
            "best_params": best_info["best_params"],
        },
        "validation_ranking": selection["validation_table"].to_dict(orient="records"),
        "calibration_choice": {
            "used_calibrated_model": calibration["use_calibrated"],
            "uncalibrated_validation_metrics": calibration["uncalibrated_validation_metrics"],
            "calibrated_validation_metrics": calibration["calibrated_validation_metrics"],
        },
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_cm,
        "test_classification_report": test_report,
    }

    with open(output / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Saved bundle to: {output / 'msme_model_bundle.joblib'}")
    print("\nValidation ranking:")
    print(selection["validation_table"].to_string(index=False))
    print("\nChosen model:")
    print(f"  {best_name} | calibrated={calibration['use_calibrated']}")
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nTest classification report:")
    print(test_report)


if __name__ == "__main__":
    # Edit the paths below if needed.
    main(
        csv_path="/mnt/data/final_dataset.csv",
        output_dir="/mnt/data/msme_outputs",
        n_iter=20,
    )
