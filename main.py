from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ARTIFACT_DIR = Path("models")


@dataclass
class TrainResult:
    module_name: str
    model_path: Path
    metrics_path: Path
    metrics: dict[str, Any]


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("--hidden-layers cannot be empty.")
    layers = tuple(int(x) for x in parts)
    if any(layer <= 0 for layer in layers):
        raise ValueError("All hidden layer sizes must be positive integers.")
    return layers


def _read_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def _detect_timestamp_col(df: pd.DataFrame) -> str | None:
    keywords = ("timestamp", "datetime", "date", "time")
    for col in df.columns:
        col_norm = _normalize_name(col)
        if any(key in col_norm for key in keywords):
            return col
    return None


def _binary_like(series: pd.Series) -> bool:
    values = set(pd.Series(series).dropna().unique().tolist())
    if not values:
        return False
    normalized = set()
    for value in values:
        if isinstance(value, str):
            val = value.strip().lower()
            if val in {"yes", "true", "y", "attack", "leak", "burst", "potable"}:
                normalized.add(1)
            elif val in {"no", "false", "n", "normal", "safe", "not potable"}:
                normalized.add(0)
            else:
                return False
        else:
            try:
                num = float(value)
            except Exception:
                return False
            if num in {0.0, 1.0, -1.0}:
                normalized.add(num)
            else:
                return False
    return normalized.issubset({-1.0, 0.0, 1.0})


def _to_binary(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> float:
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


def _detect_target_col(
    df: pd.DataFrame,
    preferred_keywords: list[str],
    force_binary: bool = False,
) -> str:
    lowered = {_normalize_name(col): col for col in df.columns}

    for key in preferred_keywords:
        for col_norm, col in lowered.items():
            if key in col_norm:
                if force_binary and not _binary_like(df[col]):
                    continue
                return col

    if force_binary:
        for col in df.columns:
            if _binary_like(df[col]):
                return col
        raise ValueError("Could not infer a binary target column. Please pass --target-col explicitly.")

    raise ValueError("Could not infer target column. Please pass --target-col explicitly.")


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    for col in datetime_cols:
        # Convert datetime to Unix seconds so tabular models can consume time fields.
        X[col] = (pd.to_datetime(X[col], errors="coerce").astype("int64") // 10**9).astype("float64")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def _save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _classification_cv_summary(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return {
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "f1_mean": float(np.mean(f1)),
        "f1_std": float(np.std(f1)),
        "roc_auc_mean": float(np.mean(auc)),
        "roc_auc_std": float(np.std(auc)),
    }


def _regression_cv_summary(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int) -> dict[str, float]:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    r2 = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    neg_rmse = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    neg_mae = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
    rmse = -neg_rmse
    mae = -neg_mae
    return {
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "mae_mean": float(np.mean(mae)),
        "mae_std": float(np.std(mae)),
    }


def _oversample_minority(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    class_counts = y.value_counts()
    if len(class_counts) != 2:
        return X, y

    majority_class = int(class_counts.idxmax())
    minority_class = int(class_counts.idxmin())

    majority_idx = y[y == majority_class].index
    minority_idx = y[y == minority_class].index
    if len(minority_idx) == 0 or len(majority_idx) == len(minority_idx):
        return X, y

    sampled_minority_idx = np.random.choice(minority_idx.to_numpy(), size=len(majority_idx), replace=True)
    final_idx = np.concatenate([majority_idx.to_numpy(), sampled_minority_idx])
    np.random.shuffle(final_idx)

    return X.loc[final_idx], y.loc[final_idx]


def _train_classifier_module(
    module_name: str,
    df: pd.DataFrame,
    target_col: str,
    artifact_dir: Path,
    hidden_layer_sizes: tuple[int, ...],
    learning_rate_init: float,
    alpha: float,
    max_iter: int,
    solver: str,
    balance_classes: bool,
    optimize_threshold: bool,
    threshold_metric: str,
    cv_folds: int,
    test_size: float = 0.2,
) -> TrainResult:
    y = _to_binary(df[target_col])
    X = df.drop(columns=[target_col]).copy()

    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx].astype(int)

    if y.nunique() < 2:
        raise ValueError(f"Target column for {module_name} has only one class after cleaning.")

    preprocessor = _build_preprocessor(X)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        solver=solver,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    decision_threshold = 0.5

    # Optional threshold tuning: split train into fit/validation, tune threshold on validation,
    # then refit on full training split for final test evaluation.
    if optimize_threshold and hasattr(pipeline, "predict_proba"):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )
        X_fit_bal, y_fit_bal = (X_fit, y_fit)
        if balance_classes:
            X_fit_bal, y_fit_bal = _oversample_minority(X_fit, y_fit)
        pipeline.fit(X_fit_bal, y_fit_bal)

        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        candidates = np.arange(0.20, 0.81, 0.02)
        best_score = -1.0
        for th in candidates:
            y_val_pred = (y_val_prob >= th).astype(int)
            if threshold_metric == "precision":
                score = precision_score(y_val, y_val_pred, zero_division=0)
            elif threshold_metric == "recall":
                score = recall_score(y_val, y_val_pred, zero_division=0)
            else:
                score = f1_score(y_val, y_val_pred, zero_division=0)
            if score > best_score:
                best_score = float(score)
                decision_threshold = float(th)

    X_train_bal, y_train_bal = (X_train, y_train)
    if balance_classes:
        X_train_bal, y_train_bal = _oversample_minority(X_train, y_train)
    pipeline.fit(X_train_bal, y_train_bal)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= decision_threshold).astype(int)
    else:
        y_pred = pipeline.predict(X_test)
        y_prob = None

    metrics = {
        "module": module_name,
        "samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "target_col": target_col,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "params": {
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate_init": learning_rate_init,
            "alpha": alpha,
            "max_iter": max_iter,
            "solver": solver,
            "balance_classes": balance_classes,
            "optimize_threshold": optimize_threshold,
            "threshold_metric": threshold_metric,
            "decision_threshold": decision_threshold,
            "test_size": test_size,
        },
    }

    if cv_folds >= 2:
        metrics["cv"] = _classification_cv_summary(pipeline, X_train, y_train, cv_folds=cv_folds)

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{module_name}_model.joblib"
    metrics_path = artifact_dir / f"{module_name}_metrics.json"

    joblib.dump(pipeline, model_path)
    _save_metrics(metrics, metrics_path)

    return TrainResult(module_name, model_path, metrics_path, metrics)


def _train_demand_module(
    df: pd.DataFrame,
    target_col: str,
    artifact_dir: Path,
    hidden_layer_sizes: tuple[int, ...],
    learning_rate_init: float,
    alpha: float,
    max_iter: int,
    solver: str,
    cv_folds: int,
    timestamp_col: str | None = None,
    test_size: float = 0.2,
) -> TrainResult:
    work = df.copy()

    if timestamp_col:
        work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
        work = work.sort_values(by=timestamp_col)
        work["hour"] = work[timestamp_col].dt.hour
        work["dayofweek"] = work[timestamp_col].dt.dayofweek
        work["month"] = work[timestamp_col].dt.month

    for lag in (1, 2, 3):
        work[f"{target_col}_lag_{lag}"] = work[target_col].shift(lag)

    work = work.dropna(subset=[target_col]).copy()
    work = work.dropna().copy()

    y = pd.to_numeric(work[target_col], errors="coerce")
    X = work.drop(columns=[target_col]).copy()

    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(X) < 50:
        raise ValueError("Demand dataset has too few usable rows after lag feature generation.")

    preprocessor = _build_preprocessor(X)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        solver=solver,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "module": "demand",
        "samples": int(len(df)),
        "usable_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "target_col": target_col,
        "timestamp_col": timestamp_col,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "params": {
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate_init": learning_rate_init,
            "alpha": alpha,
            "max_iter": max_iter,
            "solver": solver,
            "test_size": test_size,
        },
    }

    if cv_folds >= 2:
        metrics["cv"] = _regression_cv_summary(pipeline, X_train, y_train, cv_folds=cv_folds)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "demand_model.joblib"
    metrics_path = artifact_dir / "demand_metrics.json"

    joblib.dump(pipeline, model_path)
    _save_metrics(metrics, metrics_path)

    return TrainResult("demand", model_path, metrics_path, metrics)


def train_demand(
    csv_path: str,
    artifact_dir: Path,
    target_col: str | None = None,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    learning_rate_init: float = 1e-3,
    alpha: float = 1e-4,
    max_iter: int = 400,
    solver: str = "adam",
    cv_folds: int = 0,
    test_size: float = 0.2,
) -> TrainResult:
    df = _read_csv(csv_path)

    if target_col is None:
        target_col = _detect_target_col(
            df,
            preferred_keywords=["demand", "consumption", "usage", "inlet", "outlet", "flow", "volume"],
            force_binary=False,
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    timestamp_col = _detect_timestamp_col(df)
    return _train_demand_module(
        df,
        target_col,
        artifact_dir,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        solver=solver,
        cv_folds=cv_folds,
        timestamp_col=timestamp_col,
        test_size=test_size,
    )


def train_distribution(
    csv_path: str,
    artifact_dir: Path,
    target_col: str | None = None,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    learning_rate_init: float = 1e-3,
    alpha: float = 1e-4,
    max_iter: int = 300,
    solver: str = "adam",
    balance_classes: bool = False,
    optimize_threshold: bool = False,
    threshold_metric: str = "f1",
    cv_folds: int = 0,
    test_size: float = 0.2,
) -> TrainResult:
    df = _read_csv(csv_path)

    if target_col is None:
        target_col = _detect_target_col(
            df,
            preferred_keywords=["attack", "anomaly", "label", "status", "target", "class"],
            force_binary=True,
        )

    return _train_classifier_module(
        "distribution",
        df,
        target_col,
        artifact_dir,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        solver=solver,
        balance_classes=balance_classes,
        optimize_threshold=optimize_threshold,
        threshold_metric=threshold_metric,
        cv_folds=cv_folds,
        test_size=test_size,
    )


def train_leak(
    csv_path: str,
    artifact_dir: Path,
    target_col: str | None = None,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    learning_rate_init: float = 1e-3,
    alpha: float = 1e-4,
    max_iter: int = 300,
    solver: str = "adam",
    balance_classes: bool = False,
    optimize_threshold: bool = False,
    threshold_metric: str = "f1",
    cv_folds: int = 0,
    test_size: float = 0.2,
) -> TrainResult:
    df = _read_csv(csv_path)

    if target_col is None:
        leak_col = None
        burst_col = None
        for col in df.columns:
            col_norm = _normalize_name(col)
            if "leak" in col_norm:
                leak_col = col
            if "burst" in col_norm:
                burst_col = col

        if leak_col and burst_col:
            merged = pd.concat([_to_binary(df[leak_col]), _to_binary(df[burst_col])], axis=1).max(axis=1)
            df = df.copy()
            df["leak_or_burst"] = merged
            target_col = "leak_or_burst"
        elif leak_col:
            target_col = leak_col
        elif burst_col:
            target_col = burst_col
        else:
            target_col = _detect_target_col(
                df,
                preferred_keywords=["leak", "burst", "anomaly", "label", "target"],
                force_binary=True,
            )

    return _train_classifier_module(
        "leak",
        df,
        target_col,
        artifact_dir,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        solver=solver,
        balance_classes=balance_classes,
        optimize_threshold=optimize_threshold,
        threshold_metric=threshold_metric,
        cv_folds=cv_folds,
        test_size=test_size,
    )


def train_quality(
    csv_path: str,
    artifact_dir: Path,
    target_col: str | None = None,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    learning_rate_init: float = 1e-3,
    alpha: float = 1e-4,
    max_iter: int = 300,
    solver: str = "adam",
    balance_classes: bool = False,
    optimize_threshold: bool = False,
    threshold_metric: str = "f1",
    cv_folds: int = 0,
    test_size: float = 0.2,
) -> TrainResult:
    df = _read_csv(csv_path)

    if target_col is None:
        target_col = _detect_target_col(
            df,
            preferred_keywords=["potability", "potable", "quality", "label", "target"],
            force_binary=True,
        )

    return _train_classifier_module(
        "quality",
        df,
        target_col,
        artifact_dir,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        solver=solver,
        balance_classes=balance_classes,
        optimize_threshold=optimize_threshold,
        threshold_metric=threshold_metric,
        cv_folds=cv_folds,
        test_size=test_size,
    )


def integrated_decision(
    demand_score: float,
    distribution_risk: float,
    leak_probability: float,
    quality_probability: float,
) -> dict[str, Any]:
    # quality_probability is probability of safe water in [0, 1]
    # distribution_risk and leak_probability are risk probabilities in [0, 1]
    if quality_probability < 0.50:
        rule_id = "R1_QUALITY_CRITICAL"
        action = "HOLD_SUPPLY_AND_TRIGGER_TREATMENT"
        priority = "critical"
        rationale = "Safe water probability fell below 0.50, so quality risk is prioritized over all other conditions."
    elif leak_probability >= 0.70:
        rule_id = "R2_LEAK_HIGH"
        action = "ISOLATE_LEAK_ZONE_AND_DISPATCH_MAINTENANCE"
        priority = "high"
        rationale = "Leak probability is at or above 0.70, indicating likely physical loss and urgent maintenance need."
    elif distribution_risk >= 0.70:
        rule_id = "R3_DISTRIBUTION_RISK_HIGH"
        action = "REDUCE_PRESSURE_AND_REROUTE_FLOW"
        priority = "high"
        rationale = "Distribution risk is at or above 0.70, so pressure control and rerouting are triggered."
    elif demand_score >= 0.75:
        rule_id = "R4_DEMAND_SURGE"
        action = "INCREASE_SUPPLY_TO_HIGH_DEMAND_ZONES"
        priority = "medium"
        rationale = "Demand score is high (>= 0.75), so the system increases supply to expected high-usage zones."
    else:
        rule_id = "R5_NORMAL"
        action = "NORMAL_OPERATION"
        priority = "low"
        rationale = "No critical quality, leak, distribution, or demand surge rule was triggered."

    return {
        "rule_id": rule_id,
        "rationale": rationale,
        "action": action,
        "priority": priority,
        "inputs": {
            "demand_score": demand_score,
            "distribution_risk": distribution_risk,
            "leak_probability": leak_probability,
            "quality_probability": quality_probability,
        },
    }


def _print_result(result: TrainResult) -> None:
    print(f"\n[{result.module_name.upper()}]")
    print(f"Model saved:   {result.model_path}")
    print(f"Metrics saved: {result.metrics_path}")
    print(json.dumps(result.metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Water Distribution ANN Pipeline")
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR), help="Folder to save trained models and metrics")
    parser.add_argument("--hidden-layers", type=str, default="128,64", help="Comma-separated hidden layer sizes, e.g. 128,64")
    parser.add_argument("--learning-rate-init", type=float, default=1e-3, help="Initial learning rate for MLP")
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--max-iter-clf", type=int, default=300, help="Max iterations for classification modules")
    parser.add_argument("--max-iter-reg", type=int, default=400, help="Max iterations for demand regression module")
    parser.add_argument("--solver", type=str, default="adam", choices=["adam", "lbfgs", "sgd"], help="MLP solver")
    parser.add_argument("--balance-classes", action="store_true", help="Use balanced sample weights for classification modules")
    parser.add_argument("--optimize-threshold", action="store_true", help="Tune classification decision threshold on validation split")
    parser.add_argument("--threshold-metric", type=str, default="f1", choices=["f1", "recall", "precision"], help="Metric to optimize threshold")
    parser.add_argument("--cv-folds", type=int, default=0, help="Number of CV folds (0 disables CV)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Train-test split ratio")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("train-demand", help="Train demand forecasting module")
    p1.add_argument("--csv", required=True, help="Path to demand CSV")
    p1.add_argument("--target-col", default=None)

    p2 = subparsers.add_parser("train-distribution", help="Train distribution risk module")
    p2.add_argument("--csv", required=True, help="Path to distribution/BATADAL CSV")
    p2.add_argument("--target-col", default=None)

    p3 = subparsers.add_parser("train-leak", help="Train leak and burst detection module")
    p3.add_argument("--csv", required=True, help="Path to leak CSV")
    p3.add_argument("--target-col", default=None)

    p4 = subparsers.add_parser("train-quality", help="Train quality potability module")
    p4.add_argument("--csv", required=True, help="Path to quality CSV")
    p4.add_argument("--target-col", default=None)

    p5 = subparsers.add_parser("train-all", help="Train all four modules")
    p5.add_argument("--demand-csv", required=True)
    p5.add_argument("--distribution-csv", required=True)
    p5.add_argument("--leak-csv", required=True)
    p5.add_argument("--quality-csv", required=True)
    p5.add_argument("--demand-target-col", default=None)
    p5.add_argument("--distribution-target-col", default=None)
    p5.add_argument("--leak-target-col", default=None)
    p5.add_argument("--quality-target-col", default=None)

    p6 = subparsers.add_parser("recommend", help="Run integrated decision engine")
    p6.add_argument("--demand-score", type=float, required=True, help="Normalized demand score in [0, 1]")
    p6.add_argument("--distribution-risk", type=float, required=True, help="Distribution risk in [0, 1]")
    p6.add_argument("--leak-probability", type=float, required=True, help="Leak probability in [0, 1]")
    p6.add_argument("--quality-probability", type=float, required=True, help="Safe water probability in [0, 1]")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    hidden_layers = _parse_hidden_layers(args.hidden_layers)

    if args.command == "train-demand":
        result = train_demand(
            args.csv,
            artifact_dir,
            args.target_col,
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=args.learning_rate_init,
            alpha=args.alpha,
            max_iter=args.max_iter_reg,
            solver=args.solver,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
        )
        _print_result(result)
        return

    if args.command == "train-distribution":
        result = train_distribution(
            args.csv,
            artifact_dir,
            args.target_col,
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=args.learning_rate_init,
            alpha=args.alpha,
            max_iter=args.max_iter_clf,
            solver=args.solver,
            balance_classes=args.balance_classes,
            optimize_threshold=args.optimize_threshold,
            threshold_metric=args.threshold_metric,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
        )
        _print_result(result)
        return

    if args.command == "train-leak":
        result = train_leak(
            args.csv,
            artifact_dir,
            args.target_col,
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=args.learning_rate_init,
            alpha=args.alpha,
            max_iter=args.max_iter_clf,
            solver=args.solver,
            balance_classes=args.balance_classes,
            optimize_threshold=args.optimize_threshold,
            threshold_metric=args.threshold_metric,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
        )
        _print_result(result)
        return

    if args.command == "train-quality":
        result = train_quality(
            args.csv,
            artifact_dir,
            args.target_col,
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=args.learning_rate_init,
            alpha=args.alpha,
            max_iter=args.max_iter_clf,
            solver=args.solver,
            balance_classes=args.balance_classes,
            optimize_threshold=args.optimize_threshold,
            threshold_metric=args.threshold_metric,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
        )
        _print_result(result)
        return

    if args.command == "train-all":
        results = [
            train_demand(
                args.demand_csv,
                artifact_dir,
                args.demand_target_col,
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=args.learning_rate_init,
                alpha=args.alpha,
                max_iter=args.max_iter_reg,
                solver=args.solver,
                cv_folds=args.cv_folds,
                test_size=args.test_size,
            ),
            train_distribution(
                args.distribution_csv,
                artifact_dir,
                args.distribution_target_col,
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=args.learning_rate_init,
                alpha=args.alpha,
                max_iter=args.max_iter_clf,
                solver=args.solver,
                balance_classes=args.balance_classes,
                optimize_threshold=args.optimize_threshold,
                threshold_metric=args.threshold_metric,
                cv_folds=args.cv_folds,
                test_size=args.test_size,
            ),
            train_leak(
                args.leak_csv,
                artifact_dir,
                args.leak_target_col,
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=args.learning_rate_init,
                alpha=args.alpha,
                max_iter=args.max_iter_clf,
                solver=args.solver,
                balance_classes=args.balance_classes,
                optimize_threshold=args.optimize_threshold,
                threshold_metric=args.threshold_metric,
                cv_folds=args.cv_folds,
                test_size=args.test_size,
            ),
            train_quality(
                args.quality_csv,
                artifact_dir,
                args.quality_target_col,
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=args.learning_rate_init,
                alpha=args.alpha,
                max_iter=args.max_iter_clf,
                solver=args.solver,
                balance_classes=args.balance_classes,
                optimize_threshold=args.optimize_threshold,
                threshold_metric=args.threshold_metric,
                cv_folds=args.cv_folds,
                test_size=args.test_size,
            ),
        ]
        for result in results:
            _print_result(result)
        return

    if args.command == "recommend":
        decision = integrated_decision(
            demand_score=args.demand_score,
            distribution_risk=args.distribution_risk,
            leak_probability=args.leak_probability,
            quality_probability=args.quality_probability,
        )
        print(json.dumps(decision, indent=2))
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
