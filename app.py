from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from main import integrated_decision


st.set_page_config(page_title="Smart Water Distribution Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
METRIC_FILES = {
    "Demand": MODELS_DIR / "demand_metrics.json",
    "Distribution": MODELS_DIR / "distribution_metrics.json",
    "Leak": MODELS_DIR / "leak_metrics.json",
    "Quality": MODELS_DIR / "quality_metrics.json",
}
MODEL_FILES = {
    "Demand": MODELS_DIR / "demand_model.joblib",
    "Distribution": MODELS_DIR / "distribution_model.joblib",
    "Leak": MODELS_DIR / "leak_model.joblib",
    "Quality": MODELS_DIR / "quality_model.joblib",
}


def load_metrics() -> dict[str, dict]:
    metrics = {}
    for name, path in METRIC_FILES.items():
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                metrics[name] = json.load(f)
    return metrics


@st.cache_resource
def load_models() -> tuple[dict[str, object], dict[str, str]]:
    models = {}
    errors: dict[str, str] = {}
    for name, path in MODEL_FILES.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        else:
            errors[name] = "Model file is missing"
    return models, errors


@st.cache_data
def load_reference_data() -> dict[str, pd.DataFrame]:
    data = {}
    candidates = {
        "Demand": DATA_DIR / "demand" / "netbase_inlet-outlet-cont_logged_user_April2018.csv",
        "Distribution": DATA_DIR / "distribution" / "training_dataset_2.csv",
        "Leak": DATA_DIR / "leak" / "location_aware_gis_leakage_dataset.csv",
        "Quality": DATA_DIR / "quality" / "water_potability.csv",
    }
    for name, path in candidates.items():
        if path.exists():
            data[name] = pd.read_csv(path)
    return data


def _build_base_input_row(df: pd.DataFrame, target_col: str | None) -> dict[str, object]:
    work = df.copy()
    if target_col and target_col in work.columns:
        work = work.drop(columns=[target_col])

    row: dict[str, object] = {}
    for col in work.columns:
        series = work[col]
        if pd.api.types.is_numeric_dtype(series):
            row[col] = float(series.dropna().median()) if not series.dropna().empty else 0.0
        else:
            mode = series.dropna().mode()
            row[col] = mode.iloc[0] if not mode.empty else ""
    return row


def _demand_feature_frame(demand_df: pd.DataFrame, target_col: str, timestamp_col: str) -> pd.DataFrame:
    work = demand_df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.sort_values(by=timestamp_col)
    work["hour"] = work[timestamp_col].dt.hour
    work["dayofweek"] = work[timestamp_col].dt.dayofweek
    work["month"] = work[timestamp_col].dt.month
    # Keep timestamp feature numeric so inference input never mixes datetime and float dtypes.
    work[timestamp_col] = (work[timestamp_col].astype("int64") // 10**9).astype("float64")
    for lag in (1, 2, 3):
        work[f"{target_col}_lag_{lag}"] = pd.to_numeric(work[target_col], errors="coerce").shift(lag)
    work = work.dropna(subset=[target_col]).dropna()
    return work.drop(columns=[target_col])


def _to_binary_local(series: pd.Series) -> pd.Series:
    def convert(value: object) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            val = value.strip().lower()
            if val in {"yes", "true", "y", "attack", "leak", "burst", "potable", "fail", "failure"}:
                return 1.0
            if val in {"no", "false", "n", "normal", "safe", "not potable", "ok", "healthy"}:
                return 0.0
        try:
            num = float(value)
        except Exception:
            return np.nan
        if num == -1.0:
            return 1.0
        if num in {0.0, 1.0}:
            return num
        return np.nan

    return series.map(convert)


def _prepare_binary_eval_inputs(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    y = _to_binary_local(df[target_col])
    X = df.drop(columns=[target_col]).copy()
    valid_idx = y.notna()
    return X.loc[valid_idx], y.loc[valid_idx].astype(int)


def _get_eval_inputs(
    df: pd.DataFrame,
    target_col: str,
    module_metrics: dict,
    scope: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X_all, y_all = _prepare_binary_eval_inputs(df, target_col)
    if scope.startswith("Official Held-out"):
        test_size = float(module_metrics.get("params", {}).get("test_size", 0.2))
        _, X_eval, _, y_eval = train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            random_state=42,
            stratify=y_all,
        )
        return X_eval, y_eval
    return X_all, y_all


def render_classification_diagnostics(metrics: dict[str, dict]) -> None:
    st.subheader("Classification Diagnostics")
    st.caption("Technical panel for model-behavior analysis. Use held-out mode for official reporting.")

    models, model_errors = load_models()
    data = load_reference_data()
    modules = ["Distribution", "Leak", "Quality"]
    available = [m for m in modules if m in models and m in data and m in metrics]

    if model_errors:
        with st.expander("Model Load Status", expanded=False):
            st.json(model_errors)

    if not available:
        st.info("Diagnostics unavailable. Ensure models, metrics, and datasets are present.")
        return

    module = st.selectbox("Select module", available, index=0)
    module_metrics = metrics[module]
    target_col = module_metrics.get("target_col")
    if not target_col or target_col not in data[module].columns:
        st.warning(f"Target column not found for {module} diagnostics.")
        return

    scope = st.radio(
        "Evaluation scope",
        [
            "Official Held-out Test (reproducible)",
            "Full Dataset (operational monitoring)",
        ],
        horizontal=True,
    )

    default_threshold = float(module_metrics.get("params", {}).get("decision_threshold", 0.5))
    threshold = st.slider("Decision threshold", 0.05, 0.95, min(max(default_threshold, 0.05), 0.95), 0.01)

    X_eval, y_eval = _get_eval_inputs(data[module], target_col, module_metrics, scope)
    try:
        y_prob = models[module].predict_proba(X_eval)[:, 1]
    except Exception as exc:
        st.error(
            "Model inference failed due to runtime compatibility mismatch. "
            "This usually happens when model files were trained with a different scikit-learn version."
        )
        st.code(f"{type(exc).__name__}: {exc}")
        st.info(
            "Fix: retrain and re-upload model files using the same versions as deployment runtime, "
            "or pin compatible runtime/library versions."
        )
        return
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_eval, y_pred))
    prec = float(precision_score(y_eval, y_pred, zero_division=0))
    rec = float(recall_score(y_eval, y_pred, zero_division=0))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_eval, y_prob))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1", f"{f1:.4f}")
    c5.metric("ROC AUC", f"{auc:.4f}")

    cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.markdown("### Confusion Matrix")
    st.dataframe(cm_df, width="stretch")

    fpr, tpr, _ = roc_curve(y_eval, y_prob)
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr}).set_index("FPR")
    precision_arr, recall_arr, _ = precision_recall_curve(y_eval, y_prob)
    pr_df = pd.DataFrame({"Recall": recall_arr, "Precision": precision_arr}).set_index("Recall")

    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.markdown("### ROC Curve")
        st.line_chart(roc_df)
    with col_pr:
        st.markdown("### Precision-Recall Curve")
        st.line_chart(pr_df)


def build_overview_table(metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for module, data in metrics.items():
        rows.append(
            {
                "Module": module,
                "Accuracy": data.get("accuracy"),
                "F1": data.get("f1"),
                "ROC AUC": data.get("roc_auc"),
                "R2": data.get("r2"),
                "RMSE": data.get("rmse"),
                "MAE": data.get("mae"),
                "Samples": data.get("samples"),
            }
        )
    return pd.DataFrame(rows)


def render_header() -> None:
    st.title("Smart Water Distribution Management System")
    st.caption("4-module ANN pipeline: Demand, Distribution, Leak, and Quality")
    with st.expander("Path and File Status", expanded=False):
        st.write(f"App folder: {BASE_DIR}")
        st.write(f"Metrics folder: {MODELS_DIR}")
        st.json({name: path.exists() for name, path in METRIC_FILES.items()})
        st.json({name: path.exists() for name, path in MODEL_FILES.items()})


def render_metrics(metrics: dict[str, dict]) -> None:
    st.subheader("Executive Model Summary")
    if not metrics:
        st.warning("No metrics found in the models folder. Train modules first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demand R2", f"{metrics.get('Demand', {}).get('r2', 0.0):.4f}")
    c2.metric("Distribution ROC-AUC", f"{metrics.get('Distribution', {}).get('roc_auc', 0.0):.4f}")
    c3.metric("Leak ROC-AUC", f"{metrics.get('Leak', {}).get('roc_auc', 0.0):.4f}")
    c4.metric("Quality ROC-AUC", f"{metrics.get('Quality', {}).get('roc_auc', 0.0):.4f}")

    st.markdown("### Official Test Metrics (from saved training run)")

    table = build_overview_table(metrics)
    st.dataframe(table, width="stretch")

    cls_plot = table[["Module", "Accuracy", "F1", "ROC AUC"]].copy()
    cls_plot = cls_plot.fillna(0.0).set_index("Module")

    reg_plot = table[["Module", "R2", "RMSE", "MAE"]].copy().set_index("Module")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Classification Metrics")
        st.bar_chart(cls_plot)
    with col2:
        st.markdown("### Demand Regression Metrics")
        st.bar_chart(reg_plot.fillna(0.0))


def render_live_decision() -> None:
    st.subheader("Live Decision From Module Parameters")

    models, model_errors = load_models()
    data = load_reference_data()
    required = {"Demand", "Distribution", "Leak", "Quality"}

    if model_errors:
        st.warning("Some models could not be loaded in this environment.")
        with st.expander("Model Load Errors", expanded=False):
            st.json(model_errors)

    if not required.issubset(set(models.keys())):
        st.warning("Some model files are missing. Train all modules before using live prediction.")
        return

    if not required.issubset(set(data.keys())):
        st.warning("Some dataset files are missing in data folder. Keep downloaded Kaggle CSVs in the project data folder.")
        return

    demand_metrics = json.load((METRIC_FILES["Demand"]).open("r", encoding="utf-8"))
    distribution_metrics = json.load((METRIC_FILES["Distribution"]).open("r", encoding="utf-8"))
    leak_metrics = json.load((METRIC_FILES["Leak"]).open("r", encoding="utf-8"))
    quality_metrics = json.load((METRIC_FILES["Quality"]).open("r", encoding="utf-8"))

    demand_target = demand_metrics["target_col"]
    demand_ts = demand_metrics["timestamp_col"]

    demand_features_df = _demand_feature_frame(data["Demand"], demand_target, demand_ts)
    demand_base = _build_base_input_row(demand_features_df, target_col=None)

    distribution_base = _build_base_input_row(data["Distribution"], distribution_metrics["target_col"])
    leak_base = _build_base_input_row(data["Leak"], leak_metrics["target_col"])
    quality_base = _build_base_input_row(data["Quality"], quality_metrics["target_col"])

    st.markdown("Use these controls to change real input parameters. Module outputs are predicted automatically.")

    scenario = st.selectbox(
        "Scenario Preset",
        ["Custom", "Normal Operation", "High Demand", "Leak Emergency", "Poor Quality Alert"],
        index=0,
    )

    demand_defaults = {
        "lag1": float(demand_base.get(f"{demand_target}_lag_1", 0.0)),
        "lag2": float(demand_base.get(f"{demand_target}_lag_2", 0.0)),
        "lag3": float(demand_base.get(f"{demand_target}_lag_3", 0.0)),
    }
    distribution_defaults = {
        "l_t1": float(distribution_base.get("L_T1", 0.0)),
        "l_t5": float(distribution_base.get("L_T5", 0.0)),
        "p_j280": float(distribution_base.get("P_J280", 0.0)),
        "f_pu1": float(distribution_base.get("F_PU1", 0.0)),
    }
    leak_defaults = {
        "pressure": float(leak_base.get("Pressure", 0.0)),
        "flow_rate": float(leak_base.get("Flow_Rate", 0.0)),
        "vibration": float(leak_base.get("Vibration", 0.0)),
        "rpm": float(leak_base.get("RPM", 0.0)),
    }
    quality_defaults = {
        "ph": float(quality_base.get("ph", 7.0)),
        "hardness": float(quality_base.get("Hardness", 200.0)),
        "solids": float(quality_base.get("Solids", 10000.0)),
        "chloramines": float(quality_base.get("Chloramines", 7.0)),
        "sulfate": float(quality_base.get("Sulfate", 300.0)),
        "conductivity": float(quality_base.get("Conductivity", 400.0)),
        "organic_carbon": float(quality_base.get("Organic_carbon", 12.0)),
        "trihalomethanes": float(quality_base.get("Trihalomethanes", 70.0)),
        "turbidity": float(quality_base.get("Turbidity", 4.0)),
    }

    if scenario == "High Demand":
        demand_defaults["lag1"] *= 1.35
        demand_defaults["lag2"] *= 1.30
        demand_defaults["lag3"] *= 1.25
    elif scenario == "Leak Emergency":
        leak_defaults["pressure"] *= 0.65
        leak_defaults["flow_rate"] *= 1.35
        leak_defaults["vibration"] *= 1.80
        leak_defaults["rpm"] *= 1.20
        distribution_defaults["p_j280"] *= 0.80
    elif scenario == "Poor Quality Alert":
        quality_defaults["ph"] = 5.8
        quality_defaults["solids"] *= 1.60
        quality_defaults["turbidity"] *= 2.10
        quality_defaults["trihalomethanes"] *= 1.50
        quality_defaults["chloramines"] *= 1.20

    if scenario != "Custom":
        st.caption(f"Preset loaded: {scenario}")

    with st.expander("Demand Parameters", expanded=True):
        lag1 = st.number_input("Demand Lag 1", value=float(demand_defaults["lag1"]))
        lag2 = st.number_input("Demand Lag 2", value=float(demand_defaults["lag2"]))
        lag3 = st.number_input("Demand Lag 3", value=float(demand_defaults["lag3"]))

    with st.expander("Distribution Parameters", expanded=False):
        l_t1 = st.number_input("L_T1", value=float(distribution_defaults["l_t1"]))
        l_t5 = st.number_input("L_T5", value=float(distribution_defaults["l_t5"]))
        p_j280 = st.number_input("P_J280", value=float(distribution_defaults["p_j280"]))
        f_pu1 = st.number_input("F_PU1", value=float(distribution_defaults["f_pu1"]))

    with st.expander("Leak Parameters", expanded=False):
        pressure = st.number_input("Pressure", value=float(leak_defaults["pressure"]))
        flow_rate = st.number_input("Flow_Rate", value=float(leak_defaults["flow_rate"]))
        vibration = st.number_input("Vibration", value=float(leak_defaults["vibration"]))
        rpm = st.number_input("RPM", value=float(leak_defaults["rpm"]))

    with st.expander("Quality Parameters", expanded=False):
        ph = st.number_input("pH", value=float(quality_defaults["ph"]))
        hardness = st.number_input("Hardness", value=float(quality_defaults["hardness"]))
        solids = st.number_input("Solids", value=float(quality_defaults["solids"]))
        chloramines = st.number_input("Chloramines", value=float(quality_defaults["chloramines"]))
        sulfate = st.number_input("Sulfate", value=float(quality_defaults["sulfate"]))
        conductivity = st.number_input("Conductivity", value=float(quality_defaults["conductivity"]))
        organic_carbon = st.number_input("Organic_carbon", value=float(quality_defaults["organic_carbon"]))
        trihalomethanes = st.number_input("Trihalomethanes", value=float(quality_defaults["trihalomethanes"]))
        turbidity = st.number_input("Turbidity", value=float(quality_defaults["turbidity"]))

    st.markdown("---")
    st.markdown("### Manual Module Overrides (Optional)")
    with st.expander("Override Predicted Module Values", expanded=False):
        override_demand = st.checkbox("Override Demand Score")
        manual_demand = st.slider("Manual Demand Score", 0.0, 1.0, 0.5, 0.01)

        override_dist = st.checkbox("Override Distribution Risk")
        manual_dist = st.slider("Manual Distribution Risk", 0.0, 1.0, 0.5, 0.01)

        override_leak = st.checkbox("Override Leak Probability")
        manual_leak = st.slider("Manual Leak Probability", 0.0, 1.0, 0.5, 0.01)

        override_quality = st.checkbox("Override Quality Probability")
        manual_quality = st.slider("Manual Quality Probability", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict and Recommend", type="primary"):
        # Demand input row
        demand_row = demand_base.copy()
        # Timestamp and calendar features stay automatic from reference data.
        demand_row[demand_ts] = float(demand_base.get(demand_ts, 0.0))
        demand_row["hour"] = int(demand_base.get("hour", 0))
        demand_row["dayofweek"] = int(demand_base.get("dayofweek", 0))
        demand_row["month"] = int(demand_base.get("month", 1))
        demand_row[f"{demand_target}_lag_1"] = float(lag1)
        demand_row[f"{demand_target}_lag_2"] = float(lag2)
        demand_row[f"{demand_target}_lag_3"] = float(lag3)
        demand_input = pd.DataFrame([demand_row], columns=demand_features_df.columns)

        # Distribution input row
        dist_row = distribution_base.copy()
        dist_row["L_T1"] = float(l_t1)
        dist_row["L_T5"] = float(l_t5)
        dist_row["P_J280"] = float(p_j280)
        dist_row["F_PU1"] = float(f_pu1)
        distribution_input = pd.DataFrame([dist_row])

        # Leak input row
        leak_row = leak_base.copy()
        leak_row["Pressure"] = float(pressure)
        leak_row["Flow_Rate"] = float(flow_rate)
        leak_row["Vibration"] = float(vibration)
        leak_row["RPM"] = float(rpm)
        leak_input = pd.DataFrame([leak_row])

        # Quality input row
        quality_row = {
            "ph": float(ph),
            "Hardness": float(hardness),
            "Solids": float(solids),
            "Chloramines": float(chloramines),
            "Sulfate": float(sulfate),
            "Conductivity": float(conductivity),
            "Organic_carbon": float(organic_carbon),
            "Trihalomethanes": float(trihalomethanes),
            "Turbidity": float(turbidity),
        }
        quality_input = pd.DataFrame([quality_row])

        try:
            demand_pred = float(models["Demand"].predict(demand_input)[0])
            demand_series = pd.to_numeric(data["Demand"][demand_target], errors="coerce")
            dmin = float(demand_series.min())
            dmax = float(demand_series.max())
            demand_score = 0.0 if np.isclose(dmax, dmin) else float(np.clip((demand_pred - dmin) / (dmax - dmin), 0.0, 1.0))

            distribution_risk = float(models["Distribution"].predict_proba(distribution_input)[0][1])
            leak_probability = float(models["Leak"].predict_proba(leak_input)[0][1])
            quality_probability = float(models["Quality"].predict_proba(quality_input)[0][1])
        except Exception as exc:
            st.error(
                "Prediction failed due to model/runtime compatibility mismatch. "
                "Please retrain models using deployment runtime versions and upload updated model files."
            )
            st.code(f"{type(exc).__name__}: {exc}")
            return

        if override_demand:
            demand_score = manual_demand
        if override_dist:
            distribution_risk = manual_dist
        if override_leak:
            leak_probability = manual_leak
        if override_quality:
            quality_probability = manual_quality

        decision = integrated_decision(
            demand_score=demand_score,
            distribution_risk=distribution_risk,
            leak_probability=leak_probability,
            quality_probability=quality_probability,
        )

        st.markdown("### Module Predictions")
        st.json(
            {
                "predicted_demand": demand_pred,
                "demand_score": demand_score,
                "distribution_risk": distribution_risk,
                "leak_probability": leak_probability,
                "quality_probability": quality_probability,
            }
        )

        st.markdown("### Recommended Action")
        st.success(f"Action: {decision['action']}")
        st.info(f"Priority: {decision['priority']}")
        st.markdown(f"Rule Triggered: {decision['rule_id']}")
        st.write(f"Reason: {decision['rationale']}")


def main() -> None:
    render_header()
    metrics = load_metrics()
    view = st.sidebar.radio(
        "Dashboard View",
        [
            "Overview",
            "Recommendation",
            "Advanced Diagnostics",
        ],
    )

    if view == "Overview":
        render_metrics(metrics)
        st.info("Overview shows official test metrics from training artifacts.")
    elif view == "Recommendation":
        render_live_decision()
    else:
        render_classification_diagnostics(metrics)


if __name__ == "__main__":
    main()
