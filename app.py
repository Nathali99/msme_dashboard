from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


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

LAST_YEAR_REVENUE_COL = "Total_Revenue_one_years_before"
LAST_YEAR_EXPENDITURE_COL = "Total_expenditure_one_years_before"
LAST_YEAR_STAFF_COL = "Staff_one_years_before"


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
    
    features["Loss_years"] = (profit < 0).sum(axis=1)
    return features


def parse_optional_float(value: str) -> float:
    value = str(value).strip()
    if value == "":
        return np.nan
    return float(value)


@st.cache_resource
def load_bundle(bundle_path: str | Path) -> Dict:
    return joblib.load(bundle_path)


def optional_numeric_input(label: str, help_text: str = "", key: str | None = None) -> float:
    raw = st.text_input(label, value="", help=help_text, key=key)
    try:
        return parse_optional_float(raw)
    except ValueError:
        st.error(f"{label}: enter a number or leave it blank.")
        st.stop()


def format_display_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "Missing"
    return f"{float(value):.{decimals}f}"


def status_label(pred_value: int) -> str:
    return "Failed" if int(pred_value) == 1 else "Active"


def get_user_inputs() -> tuple[pd.DataFrame, bool]:
    with st.sidebar.form("business_inputs"):
        st.subheader("Business details")

        open_year = optional_numeric_input("Open year", help_text="Leave blank if unknown.")

        st.markdown("**Revenue**")
        rev_four = optional_numeric_input("Revenue four years before")
        rev_three = optional_numeric_input("Revenue three years before")
        rev_two = optional_numeric_input("Revenue two years before")
        rev_one = optional_numeric_input("Revenue one year before")

        st.markdown("**Expenditure**")
        exp_four = optional_numeric_input("Expenditure four years before")
        exp_three = optional_numeric_input("Expenditure three years before")
        exp_two = optional_numeric_input("Expenditure two years before")
        exp_one = optional_numeric_input("Expenditure one year before")

        st.markdown("**Staff**")
        staff_four = optional_numeric_input("Staff four years before")
        staff_three = optional_numeric_input("Staff three years before")
        staff_two = optional_numeric_input("Staff two years before")
        staff_one = optional_numeric_input("Staff one year before")

        submitted = st.form_submit_button("Predict business status")

    raw_df = pd.DataFrame(
        {
            "Open_Year": [open_year],
            "Staff_four_years_before": [staff_four],
            "Staff_three_years_before": [staff_three],
            "Staff_two_years_before": [staff_two],
            "Staff_one_years_before": [staff_one],
            "Total_expenditure_four_years_before": [exp_four],
            "Total_expenditure_three_years_before": [exp_three],
            "Total_expenditure_two_years_before": [exp_two],
            "Total_expenditure_one_years_before": [exp_one],
            "Total_Revenue_four_years_before": [rev_four],
            "Total_Revenue_three_years_before": [rev_three],
            "Total_Revenue_two_years_before": [rev_two],
            "Total_Revenue_one_years_before": [rev_one],
        }
    )
    return raw_df, submitted


def get_scenario_reference_inputs(raw_df: pd.DataFrame) -> Dict[str, float]:
    st.markdown("**Reference amounts for zero or missing last-year values**")
    st.caption(
        "If a last-year value is zero or missing, the multiplier uses the reference amount below as the scenario base. "
        "If the original last-year value is nonzero, the original value is used."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        revenue_ref = optional_numeric_input(
            "Revenue reference amount",
            help_text="Used only when last-year revenue is zero or missing.",
            key="revenue_reference_input",
        )
    with c2:
        expenditure_ref = optional_numeric_input(
            "Expenditure reference amount",
            help_text="Used only when last-year expenditure is zero or missing.",
            key="expenditure_reference_input",
        )
    with c3:
        staff_ref = optional_numeric_input(
            "Staff reference amount",
            help_text="Used only when last-year staff is zero or missing.",
            key="staff_reference_input",
        )

    return {
        "revenue": revenue_ref,
        "expenditure": expenditure_ref,
        "staff": staff_ref,
    }


def apply_multiplier_with_reference(
    original_value: float,
    multiplier: float,
    reference_value: float,
    round_to_int: bool = False,
) -> Tuple[float, float, str]:
    if pd.notna(original_value) and float(original_value) != 0.0:
        scenario_base = float(original_value)
        base_source = "Original"
    elif pd.notna(reference_value):
        scenario_base = float(reference_value)
        base_source = "Reference"
    else:
        scenario_base = np.nan
        base_source = "Missing"

    if pd.isna(scenario_base):
        return np.nan, np.nan, base_source

    new_value = float(scenario_base) * float(multiplier)
    if round_to_int:
        new_value = int(round(new_value))
        scenario_base = int(round(scenario_base))
    return float(new_value), float(scenario_base), base_source


def modify_raw_df(
    raw_df: pd.DataFrame,
    revenue_mult: float = 1.0,
    expenditure_mult: float = 1.0,
    staff_mult: float = 1.0,
    revenue_reference: float = np.nan,
    expenditure_reference: float = np.nan,
    staff_reference: float = np.nan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modified = raw_df.copy()
    summary_rows = []

    config = [
        ("Revenue", LAST_YEAR_REVENUE_COL, revenue_mult, revenue_reference, False),
        ("Expenditure", LAST_YEAR_EXPENDITURE_COL, expenditure_mult, expenditure_reference, False),
        ("Staff", LAST_YEAR_STAFF_COL, staff_mult, staff_reference, True),
    ]

    for label, col, mult, reference_value, round_to_int in config:
        original_value = pd.to_numeric(modified.at[0, col], errors="coerce")
        new_value, scenario_base, base_source = apply_multiplier_with_reference(
            original_value=original_value,
            multiplier=mult,
            reference_value=reference_value,
            round_to_int=round_to_int,
        )
        if pd.notna(new_value):
            modified.at[0, col] = new_value

        summary_rows.append(
            {
                "variable": label,
                "original_last_year_value": original_value,
                "scenario_base_value": scenario_base,
                "base_source": base_source,
                "multiplier": float(mult),
                "scenario_last_year_value": new_value,
            }
        )

    return modified, pd.DataFrame(summary_rows)


def predict_failure(bundle: Dict, features_df: pd.DataFrame) -> Dict[str, float]:
    model = bundle["model"]
    expected_cols = bundle["feature_columns"]
    X_new = features_df.reindex(columns=expected_cols)

    probs = model.predict_proba(X_new)[0]
    class_to_prob = {int(cls): float(prob) for cls, prob in zip(model.classes_, probs)}
    pred_class = int(model.predict(X_new)[0])

    p_failed = class_to_prob.get(1, 0.0)
    p_active = class_to_prob.get(0, 0.0)

    return {"predicted_class": pred_class, "p_failed": p_failed, "p_active": p_active}


def predict_from_raw(bundle: Dict, raw_df: pd.DataFrame) -> tuple[Dict[str, float], pd.DataFrame]:
    features_df = build_feature_frame(raw_df)
    result = predict_failure(bundle, features_df)
    return result, features_df
    
def scenario_result(
    bundle: Dict,
    raw_df: pd.DataFrame,
    references: Dict[str, float],
    scenario: str,
    factor: float,
) -> tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    if scenario == "Improve":
        modified, summary = modify_raw_df(
            raw_df,
            revenue_mult=factor,
            expenditure_mult=1.0 / factor,
            staff_mult=1.0 / factor,
            revenue_reference=references["revenue"],
            expenditure_reference=references["expenditure"],
            staff_reference=references["staff"],
        )
    elif scenario == "Stress":
        modified, summary = modify_raw_df(
            raw_df,
            revenue_mult=1.0 / factor,
            expenditure_mult=factor,
            staff_mult=factor,
            revenue_reference=references["revenue"],
            expenditure_reference=references["expenditure"],
            staff_reference=references["staff"],
        )
    else:
        raise ValueError(scenario)
    result, features = predict_from_raw(bundle, modified)
    return result, features, summary


def render_prediction_summary(result: Dict[str, float], model_name: str) -> None:
    c1, c2, c3 = st.columns(3)
    status_text = status_label(result["predicted_class"])

    if result["predicted_class"] == 1:
        st.error(f"Predicted business status: {status_text}")
    else:
        st.success(f"Predicted business status: {status_text}")

    c1.metric("Probability of failure", f"{result['p_failed'] * 100:.2f}%")
    c2.metric("Probability of being active", f"{result['p_active'] * 100:.2f}%")
    c3.metric("Chosen model", model_name)


def main() -> None:
    st.set_page_config(page_title="MSME Failure Risk Dashboard", layout="wide")
    st.title("MSME Failure Risk Dashboard")
    st.caption("Predict whether a business is likely to be active or failed, and test what-if scenarios using the saved model bundle.")

    default_bundle_path = Path("msme_outputs/msme_model_bundle.joblib")
    bundle_path = st.sidebar.text_input("Model bundle path", str(default_bundle_path))

    try:
        bundle = load_bundle(bundle_path)
    except FileNotFoundError:
        st.error(
            "Model bundle not found. Run the training script first, or point this dashboard to the correct msme_model_bundle.joblib file."
        )
        st.stop()

    raw_df, submitted = get_user_inputs()
    base_result, base_features = predict_from_raw(bundle, raw_df)

    tab1, tab2 = st.tabs(["Prediction", "What-if analysis"])

    with tab1:
        st.subheader("Computed features")
        st.dataframe(base_features.T.rename(columns={0: "value"}), use_container_width=True)

        if submitted:
            st.subheader("Prediction")
            render_prediction_summary(base_result, bundle.get("model_name", "Unknown"))

            st.subheader("Interpretation")
            if base_result["p_failed"] >= 0.70:
                st.write("This business looks high risk based on the current inputs.")
            elif base_result["p_failed"] >= 0.40:
                st.write("This business looks moderately risky based on the current inputs.")
            else:
                st.write("This business looks relatively stable based on the current inputs.")

            st.info(
                "Important: the prediction depends on the quality of the entered data. Blank inputs are allowed and will be handled the same way as in training."
            )
        else:
            st.info("Enter the business details in the sidebar and click 'Predict business status' to view the prediction.")

    with tab2:
        st.subheader("What-if analysis")
        st.write(
            "This section changes only the last-year revenue, expenditure, and staff inputs, then recomputes the engineered features and reruns the saved model."
        )

        references = get_scenario_reference_inputs(raw_df)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            revenue_mult = st.slider("Revenue last-year multiplier", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        with col_b:
            expenditure_mult = st.slider("Expenditure last-year multiplier", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        with col_c:
            staff_mult = st.slider("Staff last-year multiplier", min_value=0.1, max_value=10.0, value=1.0, step=1.0)

        modified_df, scenario_base_summary = modify_raw_df(
            raw_df,
            revenue_mult=revenue_mult,
            expenditure_mult=expenditure_mult,
            staff_mult=staff_mult,
            revenue_reference=references["revenue"],
            expenditure_reference=references["expenditure"],
            staff_reference=references["staff"],
        )
        modified_result, modified_features = predict_from_raw(bundle, modified_df)

        st.markdown("**Base prediction**")
        render_prediction_summary(base_result, bundle.get("model_name", "Unknown"))

        st.markdown("**Scenario prediction**")
        render_prediction_summary(modified_result, bundle.get("model_name", "Unknown"))

        delta_cols = st.columns(3)
        delta_cols[0].metric(
            "Failure probability change",
            f"{modified_result['p_failed'] * 100:.2f}%",
            delta=f"{(modified_result['p_failed'] - base_result['p_failed']) * 100:.2f} pp",
        )
        delta_cols[1].metric(
            "Scenario revenue last-year",
            format_display_number(pd.to_numeric(modified_df.at[0, LAST_YEAR_REVENUE_COL], errors="coerce")),
        )
        delta_cols[2].metric(
            "Scenario expenditure last-year",
            format_display_number(pd.to_numeric(modified_df.at[0, LAST_YEAR_EXPENDITURE_COL], errors="coerce")),
        )
        delta_cols[3].metric(
            "Scenario staff last-year",
            format_display_number(pd.to_numeric(modified_df.at[0, LAST_YEAR_STAFF_COL], errors="coerce")),
        )
        
        st.markdown("**Scenario bases used**")
        st.dataframe(scenario_base_summary, use_container_width=True)

        with st.expander("See modified computed features"):
            st.dataframe(modified_features.T.rename(columns={0: "value"}), use_container_width=True)

if __name__ == "__main__":
    main()
