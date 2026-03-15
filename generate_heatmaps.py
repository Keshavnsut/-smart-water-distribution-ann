from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "artifacts" / "heatmaps"


def to_binary(series: pd.Series) -> pd.Series:
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


def pick_numeric_frame(df: pd.DataFrame, target_col: str | None = None, max_cols: int = 16) -> pd.DataFrame:
    work = df.copy()

    if target_col and target_col in work.columns:
        work[target_col] = to_binary(work[target_col])

    for col in work.columns:
        if work[col].dtype == object:
            converted = pd.to_numeric(work[col], errors="coerce")
            if converted.notna().mean() > 0.7:
                work[col] = converted

    numeric = work.select_dtypes(include=[np.number]).copy()
    numeric = numeric.dropna(axis=1, how="all")

    if target_col and target_col in numeric.columns and numeric.shape[1] > max_cols:
        # Keep columns most correlated with target for readability.
        corr_to_target = numeric.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
        keep = corr_to_target.head(max_cols).index.tolist()
        numeric = numeric[keep]
    elif numeric.shape[1] > max_cols:
        numeric = numeric.iloc[:, :max_cols]

    return numeric.dropna()


def save_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
    if df.empty or df.shape[1] < 2:
        print(f"Skipped {title}: not enough numeric columns.")
        return

    corr = df.corr(numeric_only=True)
    size = max(8, min(18, int(0.8 * len(corr.columns) + 6)))

    plt.figure(figsize=(size, size * 0.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="rocket", square=True, cbar=True)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "quality",
            "path": BASE_DIR / "data" / "quality" / "water_potability.csv",
            "target": "Potability",
            "title": "Quality Module Feature Correlation Heatmap",
        },
        {
            "name": "leak",
            "path": BASE_DIR / "data" / "leak" / "location_aware_gis_leakage_dataset.csv",
            "target": "Leakage_Flag",
            "title": "Leak Module Feature Correlation Heatmap",
        },
        {
            "name": "distribution",
            "path": BASE_DIR / "data" / "distribution" / "training_dataset_2.csv",
            "target": "ATT_FLAG",
            "title": "Distribution Module Feature Correlation Heatmap",
        },
        {
            "name": "demand",
            "path": BASE_DIR / "data" / "demand" / "netbase_inlet-outlet-cont_logged_user_April2018.csv",
            "target": None,
            "title": "Demand Module Feature Correlation Heatmap",
        },
    ]

    for cfg in configs:
        if not cfg["path"].exists():
            print(f"Missing file: {cfg['path']}")
            continue

        df = pd.read_csv(cfg["path"])
        numeric = pick_numeric_frame(df, target_col=cfg["target"])
        out = OUTPUT_DIR / f"{cfg['name']}_correlation_heatmap.png"
        save_heatmap(numeric, cfg["title"], out)


if __name__ == "__main__":
    main()
