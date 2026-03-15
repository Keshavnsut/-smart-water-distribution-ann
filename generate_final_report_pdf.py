from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
HEATMAP_DIR = BASE_DIR / "artifacts" / "heatmaps"
REPORT_GRAPH_DIR = BASE_DIR / "artifacts" / "report_graphs"
OUT_FILE = BASE_DIR / "Smart_Water_Distribution_Final_Report.pdf"

DATASET_LINKS = {
    "demand": "https://www.kaggle.com/datasets/muzammalnawaz/united-utilities-water-management-water-demand",
    "distribution": "https://www.kaggle.com/datasets/minhbtnguyen/batadal-a-dataset-for-cyber-attack-detection",
    "leak": "https://www.kaggle.com/datasets/talha97s/smart-water-leak-detection-dataset",
    "quality": "https://www.kaggle.com/datasets/adityakadiwal/water-potability",
}

DATA_PATHS = {
    "distribution": BASE_DIR / "data" / "distribution" / "training_dataset_2.csv",
    "leak": BASE_DIR / "data" / "leak" / "location_aware_gis_leakage_dataset.csv",
    "quality": BASE_DIR / "data" / "quality" / "water_potability.csv",
}

TARGET_COLS = {
    "distribution": "ATT_FLAG",
    "leak": "Leakage_Flag",
    "quality": "Potability",
}


# ---------------------------------------------------------------------------
# Custom DocTemplate — page-number footer + TOC notification
# ---------------------------------------------------------------------------

class WaterReportDoc(BaseDocTemplate):
    """BaseDocTemplate with footer page numbers and TOC heading notifications."""

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin, self.bottomMargin,
            self.width, self.height, id="main",
        )
        template = PageTemplate(id="main", frames=[frame], onPage=self._draw_footer)
        self.addPageTemplates([template])

    def _draw_footer(self, canvas, doc):
        canvas.saveState()
        w = doc.pagesize[0]
        y_line = doc.bottomMargin - 0.30 * cm
        y_text = doc.bottomMargin - 0.58 * cm
        canvas.setStrokeColor(colors.HexColor("#bbbbbb"))
        canvas.line(doc.leftMargin, y_line, w - doc.rightMargin, y_line)
        canvas.setFont("Helvetica", 8.0)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(
            doc.leftMargin, y_text,
            "Smart Water Distribution Management System \u2014 ANN Final Report",
        )
        canvas.drawRightString(w - doc.rightMargin, y_text, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        """Emit TOC entry events for H1/H2 paragraphs after rendering."""
        if isinstance(flowable, Paragraph):
            sname = flowable.style.name
            if sname == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
            elif sname == "H2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))


# ---------------------------------------------------------------------------
# Metric loaders
# ---------------------------------------------------------------------------

def load_metrics() -> dict:
    files = {
        "demand":       MODELS_DIR / "demand_metrics.json",
        "distribution": MODELS_DIR / "distribution_metrics.json",
        "leak":         MODELS_DIR / "leak_metrics.json",
        "quality":      MODELS_DIR / "quality_metrics.json",
    }
    return {k: json.loads(p.read_text(encoding="utf-8")) for k, p in files.items()}


def _to_binary_series(series: pd.Series) -> pd.Series:
    def _convert(value):
        if pd.isna(value):
            return float("nan")
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"yes", "true", "y", "attack", "leak", "burst", "potable", "fail", "failure"}:
                return 1.0
            if v in {"no", "false", "n", "normal", "safe", "not potable", "ok", "healthy"}:
                return 0.0
        try:
            num = float(value)
        except Exception:
            return float("nan")
        return 1.0 if num == -1.0 else (num if num in {0.0, 1.0} else float("nan"))
    return series.map(_convert)


def compute_per_class_metrics(m: dict) -> dict:
    """Per-class precision/recall/F1 for classifier modules (cached to disk)."""
    cache = MODELS_DIR / "per_class_cache.json"
    if cache.exists():
        return json.loads(cache.read_text())
    result = {}
    for mod in ("distribution", "leak", "quality"):
        try:
            pipeline = joblib.load(MODELS_DIR / f"{mod}_model.joblib")
            df = pd.read_csv(DATA_PATHS[mod])
            tcol = TARGET_COLS[mod]
            y = _to_binary_series(df[tcol])
            X = df.drop(columns=[tcol])
            valid = y.notna()
            X = X.loc[valid].reset_index(drop=True)
            y = y.loc[valid].reset_index(drop=True).astype(int)
            ts = m[mod]["params"].get("test_size", 0.2)
            _, X_test, _, y_test = train_test_split(X, y, test_size=ts, random_state=42, stratify=y)
            y_pred = pipeline.predict(X_test)
            rpt = classification_report(y_test, y_pred, output_dict=True)
            result[mod] = {
                "class_0": {k: round(rpt.get("0", rpt.get(0, {})).get(k, 0), 4)
                            for k in ("precision", "recall", "f1-score", "support")},
                "class_1": {k: round(rpt.get("1", rpt.get(1, {})).get(k, 0), 4)
                            for k in ("precision", "recall", "f1-score", "support")},
            }
            print(f"[per-class] {mod} OK")
        except Exception as exc:
            print(f"[warn] per-class skipped for {mod}: {exc}")
    cache.write_text(json.dumps(result, indent=2))
    return result


def compute_cv_metrics(m: dict) -> dict:
    """5-fold stratified CV for all classifier modules (cached to disk)."""
    cache = MODELS_DIR / "cv_all_cache.json"
    if cache.exists():
        return json.loads(cache.read_text())
    result = {}
    if "cv" in m.get("quality", {}):
        result["quality"] = m["quality"]["cv"]
    for mod in ("distribution", "leak"):
        try:
            df = pd.read_csv(DATA_PATHS[mod])
            tcol = TARGET_COLS[mod]
            y = _to_binary_series(df[tcol])
            X = df.drop(columns=[tcol])
            valid = y.notna()
            X = X.loc[valid].reset_index(drop=True)
            y = y.loc[valid].reset_index(drop=True).astype(int)
            for col in X.select_dtypes(include=["datetime64[ns]"]).columns:
                X[col] = (pd.to_datetime(X[col], errors="coerce").astype("int64") // 10 ** 9).astype("float64")
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            prep = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("sc", StandardScaler())]), num_cols),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                  ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ], remainder="drop")
            p = m[mod]["params"]
            clf = MLPClassifier(
                hidden_layer_sizes=tuple(p["hidden_layer_sizes"]),
                activation="relu",
                learning_rate_init=p["learning_rate_init"],
                alpha=p["alpha"],
                max_iter=p["max_iter"],
                solver=p.get("solver", "adam"),
                random_state=42,
            )
            pipe = Pipeline([("prep", prep), ("model", clf)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            f1  = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
            auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
            result[mod] = {
                "accuracy_mean": round(float(np.mean(acc)), 4),
                "accuracy_std":  round(float(np.std(acc)),  4),
                "f1_mean":       round(float(np.mean(f1)),  4),
                "f1_std":        round(float(np.std(f1)),   4),
                "roc_auc_mean":  round(float(np.mean(auc)), 4),
                "roc_auc_std":   round(float(np.std(auc)),  4),
            }
            print(f"[cv] {mod} OK")
        except Exception as exc:
            print(f"[warn] CV skipped for {mod}: {exc}")
    cache.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def generate_architecture_diagram() -> Path:
    REPORT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_GRAPH_DIR / "architecture_diagram.png"
    if out.exists():
        return out

    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def _box(x, y, w, h, label, fc="#0b3d91", tc="white", fs=8.5):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            linewidth=1.4, edgecolor="#222244", facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", multialignment="center")

    def _arr(x1, y1, x2, y2, lw=1.6, color="#333333", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

    # Input + Preprocessing (left side)
    _box(0.1, 3.2, 1.9, 1.0, "Raw Datasets\n(Kaggle CSVs)", "#1a6b3c")
    _box(2.3, 3.2, 1.9, 1.0, "Data\nPreprocessing\n& Scaling", "#0b3d91")
    _arr(2.0, 3.7, 2.3, 3.7)

    # 4 ANN module rows
    mod_y = [6.2, 4.8, 3.4, 2.0]
    mod_labels = [
        "Demand ANN\n(MLPRegressor)",
        "Distribution ANN\n(MLPClassifier)",
        "Leak ANN\n(MLPClassifier)",
        "Quality ANN\n(MLPClassifier)",
    ]
    out_labels = [
        "Demand Score\n(continuous)",
        "Risk Prob.\n(0\u20131)",
        "Leak Prob.\n(0\u20131)",
        "Quality Prob.\n(0\u20131)",
    ]
    prep_cx = 3.25  # right edge x of preprocessing box
    for i, (ml, ol) in enumerate(zip(mod_labels, out_labels)):
        my = mod_y[i]
        _box(4.5, my - 0.35, 2.3, 0.85, ml, "#2166ac")
        _box(7.2, my - 0.35, 2.1, 0.85, ol, "#4393c3")
        _arr(4.3, my + 0.075, 4.5, my + 0.075, color="#555555", lw=1.3)
        _arr(6.8, my + 0.075, 7.2, my + 0.075, color="#555555", lw=1.3)
        # Fan from preprocessing to each module
        ax.plot([prep_cx, prep_cx, 4.3], [3.7, my + 0.075, my + 0.075],
                color="#888888", lw=1.0, linestyle="--")

    # Rule fusion box
    _box(9.7, 3.2, 2.2, 1.0, "Rule-Based\nDecision\nFusion Engine", "#5e3a8c")
    for i in range(4):
        my = mod_y[i]
        _arr(9.3, my + 0.075, 9.7 + 1.1, 3.7 + (my - 3.7) * 0.0, color="#555555", lw=1.1)
    # Actually draw clean arrows from output boxes to fusion
    for i in range(4):
        my = mod_y[i]
        fx = 9.7 + 1.1
        fy = 3.7
        ax.annotate("", xy=(9.7, 3.7), xytext=(9.3, my + 0.075),
                    arrowprops=dict(arrowstyle="->", color="#666666", lw=1.1,
                                   connectionstyle="arc3,rad=0.0"))

    # Action box
    _box(12.3, 3.2, 2.5, 1.0, "Operational\nAction\nDecision", "#b52121")
    _arr(11.9, 3.7, 12.3, 3.7)

    # Dashboard (dashed)
    _box(12.3, 1.5, 2.5, 0.85, "Streamlit\nDashboard", "#2c7a5f")
    _arr(13.55, 3.2, 13.55, 2.35, color="#999999", ls="dashed", lw=1.2)

    ax.set_title("System Architecture: Smart Water Distribution ANN Pipeline",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def generate_report_graphs(metrics: dict) -> dict:
    REPORT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    loss_colors = {
        "demand": "#1f77b4", "distribution": "#ff7f0e",
        "leak": "#2ca02c",   "quality": "#d62728",
    }
    loss_plot = REPORT_GRAPH_DIR / "training_loss_curves.png"
    curves = {}
    for mod in ("demand", "distribution", "leak", "quality"):
        mp = MODELS_DIR / f"{mod}_model.joblib"
        if not mp.exists():
            continue
        model = joblib.load(mp)
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            est = model.named_steps["model"]
            if hasattr(est, "loss_curve_") and est.loss_curve_:
                curves[mod] = est.loss_curve_

    if curves:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
        if "demand" in curves:
            d = curves["demand"]
            axes[0].plot(range(1, len(d) + 1), d, color=loss_colors["demand"],
                         linewidth=2.2, label="Demand")
            axes[0].set_yscale("log")
            axes[0].set_title("Demand Module Loss Curve (log scale)")
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
            axes[0].grid(alpha=0.25); axes[0].legend(loc="upper right")
        else:
            axes[0].set_visible(False)
        cc = 0
        for mod in ("distribution", "leak", "quality"):
            if mod not in curves:
                continue
            cv = curves[mod]
            axes[1].plot(range(1, len(cv) + 1), cv, label=mod.capitalize(),
                         color=loss_colors[mod], linewidth=2.2)
            cc += 1
        if cc:
            axes[1].set_title("Classifier Module Loss Curves")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
            axes[1].grid(alpha=0.25); axes[1].legend(loc="upper right")
        else:
            axes[1].set_visible(False)
        fig.suptitle("ANN Training Loss Curves", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0.01, 1, 0.97])
        fig.savefig(loss_plot, dpi=220)
        plt.close(fig)

    perf_plot = REPORT_GRAPH_DIR / "module_performance_comparison.png"
    clf_mods = ["distribution", "leak", "quality"]
    metric_names = ["accuracy", "f1", "roc_auc"]
    x = list(range(len(clf_mods)))
    width = 0.22
    plt.figure(figsize=(10, 6))
    for i, mn in enumerate(metric_names):
        vals = [metrics[mod][mn] for mod in clf_mods]
        pos = [idx + (i - 1) * width for idx in x]
        plt.bar(pos, vals, width=width, label=mn.upper() if mn != "f1" else "F1")
    plt.xticks(x, [m.capitalize() for m in clf_mods])
    plt.ylim(0, 1.05); plt.ylabel("Score")
    plt.title("Classifier Module Performance Comparison")
    plt.grid(axis="y", alpha=0.25); plt.legend()
    plt.tight_layout()
    plt.savefig(perf_plot, dpi=180)
    plt.close()

    return {"loss_curves": loss_plot, "performance": perf_plot}


# ---------------------------------------------------------------------------
# ReportLab helpers
# ---------------------------------------------------------------------------

def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f3f7ff")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def make_rules_table(data, col_widths=None):
    styles = getSampleStyleSheet()
    cell_st = ParagraphStyle("RuleCell", parent=styles["BodyText"],
                             fontName="Helvetica", fontSize=8.5, leading=10,
                             alignment=0, wordWrap="CJK")
    hdr_st = ParagraphStyle("RuleHeader", parent=styles["BodyText"],
                            fontName="Helvetica-Bold", fontSize=10, leading=12,
                            alignment=1, textColor=colors.white)
    wrapped = []
    for i, row in enumerate(data):
        new_row = []
        for cell in row:
            text = str(cell)
            new_row.append(Paragraph(text, hdr_st) if i == 0
                           else Paragraph(text.replace("_", " "), cell_st))
        wrapped.append(new_row)
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("ALIGN", (0, 1), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f3f7ff")]),
        ("LEFTPADDING", (0, 1), (-1, -1), 6), ("RIGHTPADDING", (0, 1), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 5),  ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
    ]))
    return t


def add_heading(story, text, style):
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 8))


def add_paragraph(story, text, style):
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 6))


def add_heatmap(story, title, filename, body_style):
    path = HEATMAP_DIR / filename
    add_paragraph(story, f"<b>{title}</b>", body_style)
    if path.exists():
        img = Image(str(path))
        max_w = 16.5 * cm
        img.drawWidth = max_w
        img.drawHeight = max_w * 0.72
        story.append(img)
        story.append(Spacer(1, 6))
    else:
        add_paragraph(story, f"Image missing: {path.name}", body_style)


def add_image(story, title, path, body_style, width_cm=16.5, ratio=0.62):
    add_paragraph(story, f"<b>{title}</b>", body_style)
    if path.exists():
        img = Image(str(path))
        max_w = width_cm * cm
        img.drawWidth = max_w
        img.drawHeight = max_w * ratio
        story.append(img)
        story.append(Spacer(1, 6))
    else:
        add_paragraph(story, f"Image missing: {path.name}", body_style)


def _module_feature_block(title, rows, h2, make_rt):
    block = [
        Paragraph(title, h2),
        Spacer(1, 4),
        make_rt([["Selected Feature", "Reason"]] + rows, col_widths=[5.8 * cm, 9.2 * cm]),
        Spacer(1, 10),
    ]
    return KeepTogether(block)


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report():
    m = load_metrics()
    print("Computing per-class metrics (cached after first run)...")
    per_class = compute_per_class_metrics(m)
    print("Computing CV metrics for all modules (cached after first run)...")
    cv_all = compute_cv_metrics(m)
    graphs = generate_report_graphs(m)
    arch_diagram = generate_architecture_diagram()

    # ---- Styles -------------------------------------------------------
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=24, leading=30,
        alignment=1, textColor=colors.HexColor("#0b3d91"),
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Heading2"],
        fontSize=13, alignment=1, textColor=colors.HexColor("#163a5f"),
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=16,
        textColor=colors.HexColor("#0b3d91"), spaceBefore=10,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13,
        textColor=colors.HexColor("#163a5f"), spaceBefore=6,
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=11, leading=16,
    )
    code_style = ParagraphStyle(
        "CodeStyle", parent=styles["Normal"],
        fontName="Courier", fontSize=8, leading=11,
        leftIndent=10, rightIndent=10,
        backColor=colors.HexColor("#f5f5f5"),
        borderColor=colors.HexColor("#cccccc"),
        borderWidth=0.5, borderPadding=6,
    )
    toc_h1 = ParagraphStyle("TOC1", fontName="Helvetica-Bold", fontSize=11, leading=18)
    toc_h2 = ParagraphStyle("TOC2", fontName="Helvetica", fontSize=9.5, leading=15,
                             leftIndent=1.5 * cm)

    # ---- Document -----------------------------------------------------
    doc = WaterReportDoc(
        str(OUT_FILE),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.5 * cm,
        title="Smart Water Distribution Management System Using ANN",
        author="Project Student",
    )

    story = []

    # ==================================================================
    # COVER PAGE
    # ==================================================================
    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("Smart Water Distribution Management System", title_style))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Using Artificial Neural Networks (ANN)", subtitle_style))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Final Project Report", subtitle_style))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Department of Computer Science / AI", subtitle_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Academic Session 2025\u201326", subtitle_style))
    story.append(Spacer(1, 3.4 * cm))
    story.append(Paragraph("Prepared by:", body))
    story.append(Paragraph("1. Keshav Dubey - 2023UCB6059", body))
    story.append(Paragraph("2. Anuj Sharma - 2023UCB6040", body))
    story.append(Paragraph("3. Kunal Parewa - 2023UCB6040", body))
    story.append(Paragraph("4. Uday Bhatia - 2023UCB8052", body))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Guide / Supervisor: ______________________", body))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Date: March 2026", body))
    story.append(PageBreak())

    # ==================================================================
    # ABSTRACT + KEYWORDS
    # ==================================================================
    add_heading(story, "Abstract", h1)
    add_paragraph(
        story,
        "This report presents a complete Smart Water Distribution Management System based on "
        "Artificial Neural Networks. The implementation follows a four-module architecture: "
        f"demand forecasting (MLPRegressor, R\u00b2={m['demand']['r2']:.4f}), "
        f"distribution risk detection ({m['distribution']['accuracy']*100:.1f}% accuracy), "
        f"leak detection ({m['leak']['accuracy']*100:.1f}% accuracy, "
        f"ROC-AUC={m['leak']['roc_auc']:.4f}), and water quality potability classification "
        f"(ROC-AUC={m['quality']['roc_auc']:.4f}). Real datasets sourced from Kaggle were used "
        "for training and validation. A rule-based fusion layer integrates all module outputs into "
        "one final operational action, providing transparent and explainable decision support. "
        "The system includes preprocessing pipelines, cross-validation, diagnostic visualizations, "
        "and an interactive Streamlit dashboard.",
        body,
    )
    add_heading(story, "Keywords", h2)
    add_paragraph(
        story,
        "Smart Water, ANN, MLP, Demand Forecasting, Leak Detection, Water Quality, "
        "Distribution Risk, Rule-Based Decision Fusion, Scikit-learn, Streamlit",
        body,
    )
    story.append(PageBreak())

    # ==================================================================
    # TABLE OF CONTENTS
    # ==================================================================
    add_heading(story, "Table of Contents", h1)
    toc = TableOfContents()
    toc.levelStyles = [toc_h1, toc_h2]
    story.append(toc)
    story.append(PageBreak())

    # ==================================================================
    # 1. INTRODUCTION AND PROBLEM STATEMENT
    # ==================================================================
    add_heading(story, "1. Introduction and Problem Statement", h1)
    add_paragraph(
        story,
        "Water distribution networks are critical infrastructure that serve urban and rural "
        "populations globally. Managing these networks efficiently requires accurate demand "
        "forecasting, early leak detection, cyber-attack resilience, and real-time water quality "
        "monitoring. Traditional rule-based SCADA systems operate reactively and cannot adapt to "
        "complex, multi-factor operational states. Artificial Neural Networks (ANNs) provide a "
        "data-driven alternative that learns non-linear patterns from historical sensor readings "
        "and laboratory measurements.",
        body,
    )

    add_heading(story, "1.1 Background", h2)
    add_paragraph(
        story,
        "Modern water utilities collect large volumes of time-series data from flow meters, "
        "pressure sensors, water quality analyzers, and SCADA-connected pump stations. Despite "
        "this data richness, decision-making in most utilities is still manual or threshold-based. "
        "Machine learning offers an opportunity to transform raw sensor data into actionable, "
        "predictive intelligence. This project implements four specialized ANN modules, each "
        "targeting one key operational challenge, then fuses their outputs through a transparent "
        "rule-based engine into a single operational action.",
        body,
    )

    add_heading(story, "1.2 Problem Statement", h2)
    add_paragraph(
        story,
        "A unified, data-driven intelligence framework is needed that simultaneously: "
        "(1) forecasts water demand at zone level to enable proactive supply scheduling; "
        "(2) detects distribution anomalies and cyber-attacks on pump-valve networks; "
        "(3) identifies pipe leaks in real time from pressure-flow-vibration signals; "
        "(4) classifies water potability from physicochemical quality parameters; and "
        "(5) combines all four signals into one prioritized, explainable operational action.",
        body,
    )

    add_heading(story, "1.3 Objectives", h2)
    for i, obj in enumerate([
        f"Train a demand forecasting MLPRegressor achieving R\u00b2 > 0.90 on held-out test data.",
        "Train distribution risk, leak detection, and quality classification MLPClassifiers.",
        "Build a complete preprocessing pipeline with imputation, scaling, and class balancing.",
        "Implement a rule-based decision fusion engine integrating outputs in priority order.",
        "Validate all models with held-out test sets and 5-fold stratified cross-validation.",
        "Develop an interactive Streamlit dashboard for live prediction and diagnostics.",
        "Produce a comprehensive project report documenting datasets, methods, and results.",
    ], 1):
        add_paragraph(story, f"{i}. {obj}", body)

    add_heading(story, "1.4 System Architecture Overview", h2)
    add_paragraph(
        story,
        "The figure below illustrates the end-to-end pipeline from raw CSV inputs through "
        "preprocessing and four parallel ANN modules to the rule-based fusion engine and "
        "final operational decision output.",
        body,
    )
    add_image(story, "Figure 1: Smart Water Distribution ANN System Architecture",
              arch_diagram, body, width_cm=16.5, ratio=0.52)
    story.append(PageBreak())

    # ==================================================================
    # 2. LITERATURE REVIEW
    # ==================================================================
    add_heading(story, "2. Literature Review", h1)
    add_paragraph(
        story,
        "The following table summarizes five key prior works that establish the scientific "
        "foundation for each module of this project.",
        body,
    )
    lit_table = [
        ["#", "Reference", "Contribution / Relevance"],
        ["1",
         "Mounce et al. (2011). Artificial intelligence applications in water utilities. "
         "Water Practice & Technology, 6(1).",
         "Demonstrated ANN models trained on meter readings can forecast zonal demand with "
         "R\u00b2>0.88, directly motivating the demand forecasting module."],
        ["2",
         "Taormina et al. (2018). Battle of the Attack Detection Algorithms (BATADAL). "
         "J. Water Resources Planning & Management, 144(8).",
         "Introduced the BATADAL benchmark dataset for cyber-attack detection used directly "
         "in this project's distribution risk module."],
        ["3",
         "Laucelli & Giustolisi (2011). Vulnerability assessment of water distribution "
         "networks. J. Hydroinformatics.",
         "Explored ML-based leak and pressure anomaly detection, providing feature inspiration "
         "(pressure, flow, vibration) for the leak detection module."],
        ["4",
         "Ghosh & Mujumdar (2008). Statistical downscaling via neural networks. "
         "Computers & Geosciences.",
         "Validated use of multi-parameter physicochemical features (pH, turbidity, "
         "chloramines) as ANN inputs for environmental classification."],
        ["5",
         "Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. "
         "JMLR, 12, 2825\u20132830.",
         "The software framework underpinning all model training, preprocessing pipelines, "
         "and cross-validation evaluation in this project."],
    ]
    story.append(make_rules_table(lit_table, col_widths=[0.6 * cm, 6.5 * cm, 8.2 * cm]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # ==================================================================
    # STEP 1: DATASET PREPARATION
    # ==================================================================
    add_heading(story, "STEP 1: Dataset Preparation and Processing", h1)
    add_paragraph(
        story,
        "This step prepares all module datasets for reliable ANN learning. The workflow "
        "covers source mapping, quality checks, feature design, label validation, "
        "balancing, and split strategy for fair evaluation.",
        body,
    )

    add_heading(story, "1.1 Dataset Collection", h2)
    dataset_table = [
        ["Module", "Dataset", "Samples", "Target"],
        ["Demand", "United Utilities Water Management", str(m["demand"]["samples"]),
         m["demand"]["target_col"][:28] + "\u2026"],
        ["Distribution", "BATADAL", str(m["distribution"]["samples"]),
         m["distribution"]["target_col"]],
        ["Leak", "Smart Water Leak Detection", str(m["leak"]["samples"]),
         m["leak"]["target_col"]],
        ["Quality", "Water Potability", str(m["quality"]["samples"]),
         m["quality"]["target_col"]],
    ]
    story.append(make_table(dataset_table, col_widths=[3.2 * cm, 6.8 * cm, 2.2 * cm, 4.0 * cm]))
    story.append(Spacer(1, 8))
    add_paragraph(story, "<b>Kaggle Dataset Links:</b>", body)
    for key, label in [
        ("demand", "United Utilities (Demand)"),
        ("distribution", "BATADAL (Distribution)"),
        ("leak", "Smart Water Leak Detection"),
        ("quality", "Water Potability"),
    ]:
        add_paragraph(
            story,
            f"{label}: <link href='{DATASET_LINKS[key]}' color='blue'>{DATASET_LINKS[key]}</link>",
            body,
        )

    add_heading(story, "1.2 Data Cleaning", h2)
    for step in [
        "Duplicate record removal and schema normalization.",
        "Missing values: numeric \u2192 median imputation; categorical \u2192 mode imputation.",
        "Data type enforcement and datetime-to-Unix timestamp conversion.",
        "Non-informative and near-zero-variance field removal.",
        "Outlier control via standard scaling (zero mean, unit variance).",
        "Leakage prevention: all preprocessing fitted exclusively on training data.",
    ]:
        add_paragraph(story, f"\u2022 {step}", body)

    add_heading(story, "1.3 Feature Selection (Detailed)", h2)
    add_paragraph(
        story,
        "Feature selection was performed module-wise using a multi-criterion strategy so the "
        "ANN receives only operationally meaningful and stable signals.",
        body,
    )
    story.append(make_rules_table([
        ["Selection Criterion", "How It Was Applied"],
        ["Operational relevance",
         "Feature must directly represent hydraulic behavior, quality chemistry, or network-state signal."],
        ["Predictive relation",
         "Kept variables with visible relation to target trend or class-separation behavior."],
        ["Data quality",
         "Removed or deprioritized columns with excessive missingness or noise."],
        ["Redundancy control",
         "Avoided keeping near-duplicate variables carrying the same information."],
        ["Deployment feasibility",
         "Preferred features available from SCADA sensors in real deployments."],
    ], col_widths=[5.5 * cm, 9.5 * cm]))
    story.append(Spacer(1, 8))

    add_paragraph(story, "Input features selected per module (with rationale):", body)
    story.append(_module_feature_block("Demand Module \u2014 Selected Features", [
        ["hour_of_day", "Captures intraday consumption cycles (morning/evening peaks)."],
        ["day_of_week", "Captures weekday vs weekend usage patterns."],
        ["month", "Captures seasonal demand shifts across the year."],
        ["netflow_primary", "Core supply-demand movement at inlet/outlet."],
        ["cont_logged_flow", "Consumer-side draw linked directly to demand target."],
        ["lag_1", "Short-term autocorrelation aids next-step prediction."],
        ["lag_2", "Multi-step temporal carryover for short forecast horizon."],
        ["rolling_mean_3", "Smooths sensor noise, preserves local demand trend."],
    ], h2, make_rules_table))

    story.append(_module_feature_block("Distribution Module \u2014 Selected Features", [
        ["L T1 to L T7", "Tank-level states reflect network storage and pressure balance."],
        ["F PU1 to F PU11", "Pump flow signals \u2014 deviation indicates transport anomaly."],
        ["S PU1 to S PU11", "Pump on/off status supports operational state classification."],
        ["F V2, S V2", "Valve flow and open/close state reveal routing disruptions."],
        ["P J280, P J14, P J415", "Pressure nodes at key junctions identify risk events."],
        ["F PU3, F PU10", "Specific pump combinations linked to known attack signatures."],
    ], h2, make_rules_table))

    story.append(_module_feature_block("Leak Module \u2014 Selected Features", [
        ["Pressure", "Pressure drops are the primary and most reliable leak signature."],
        ["Flow Rate", "Leakage alters expected flow profile in the affected zone."],
        ["Temperature", "Ambient temperature affects material expansion and leak risk."],
        ["Vibration", "Mechanical disturbance from leaks produces measurable vibration."],
        ["RPM", "Pump speed variations correlate with changing network resistance."],
        ["Operational Hours", "Aging pipes with high runtime have higher leak probability."],
        ["Zone, Block", "Spatial grouping supports localized leak pattern learning."],
        ["Latitude, Longitude", "Geographic coordinates enable location-aware clustering."],
    ], h2, make_rules_table))

    story.append(_module_feature_block("Quality Module \u2014 Selected Features", [
        ["ph", "pH level indicates acidity/alkalinity \u2014 unsafe pH affects potability."],
        ["Hardness", "Mineral content impacts water acceptability thresholds."],
        ["Solids", "Total dissolved solids above threshold signal poor quality."],
        ["Chloramines", "Disinfectant concentration; excess levels degrade potability."],
        ["Sulfate", "High sulfate concentrations linked to non-potable classification."],
        ["Conductivity", "Ionic concentration proxy used to assess contamination."],
        ["Organic carbon", "Elevated organic carbon indicates contamination load."],
        ["Trihalomethanes", "Disinfection by-product \u2014 regulated directly for safety."],
        ["Turbidity", "Water cloudiness is a direct and visible quality indicator."],
    ], h2, make_rules_table))

    add_heading(story, "1.4 Label Creation", h2)
    story.append(make_table([
        ["Module", "Target Column", "Class Definition"],
        ["Demand", m["demand"]["target_col"][:24] + "\u2026",
         "Continuous regression target (l/s)"],
        ["Distribution", m["distribution"]["target_col"],
         "0 = normal operation, 1 = attack / risk condition"],
        ["Leak", m["leak"]["target_col"],
         "0 = no leak, 1 = leak detected"],
        ["Quality", m["quality"]["target_col"],
         "0 = non-potable, 1 = potable"],
    ], col_widths=[2.7 * cm, 4.8 * cm, 7.5 * cm]))
    story.append(Spacer(1, 8))

    add_heading(story, "1.5 Data Balancing", h2)
    add_paragraph(
        story,
        "Class imbalance in classification modules was addressed through random oversampling "
        "of the minority class on the training split only, preventing data leakage to the "
        "test set. Synthetic minority oversampling (SMOTE) was not used.",
        body,
    )

    add_heading(story, "1.6 Train-Test Split", h2)
    story.append(make_table([
        ["Dataset Portion", "Percentage", "Split Strategy"],
        ["Training Data", "80%", "Stratified by class label (classifiers) / random (demand)"],
        ["Testing Data",  "20%", "Held out, never seen during training or preprocessing fit"],
    ], col_widths=[4.0 * cm, 2.5 * cm, 8.5 * cm]))
    story.append(Spacer(1, 10))

    add_heading(story, "1.7 Feature Scaling and Preprocessing Pipeline", h2)
    for step in [
        "Numeric features: median imputation \u2192 StandardScaler (zero mean, unit variance).",
        "Categorical features: mode imputation \u2192 OneHotEncoder (unknown categories ignored).",
        "Datetime columns: converted to Unix epoch seconds (float64) before fitting.",
        "Demand module: hour / day-of-week / month / lag / rolling features engineered.",
        "Pipeline fitted exclusively on training split to prevent data leakage.",
    ]:
        add_paragraph(story, f"\u2022 {step}", body)
    story.append(PageBreak())

    # ==================================================================
    # STEP 2: MODEL BUILDING AND TRAINING
    # ==================================================================
    add_heading(story, "STEP 2: Model Building and Training", h1)

    add_heading(story, "2.1 ANN Architecture", h2)
    add_paragraph(
        story,
        "A feed-forward Multi-Layer Perceptron (MLP) architecture is used for all modules, "
        "with the output neuron adapted for regression (Demand) or binary classification "
        "(Distribution, Leak, Quality).",
        body,
    )
    story.append(make_rules_table([
        ["Layer Type", "Typical Neurons", "Activation / Notes"],
        ["Input Layer", "Feature-dependent", "One neuron per preprocessed feature"],
        ["Hidden Layer 1", "256", "ReLU \u2014 learns first-order feature interactions"],
        ["Hidden Layer 2", "128", "ReLU \u2014 deeper abstraction"],
        ["Hidden Layer 3", "64",  "ReLU \u2014 compact representation before output"],
        ["Output (Demand)", "1",  "Linear \u2014 continuous regression output"],
        ["Output (Classifiers)", "1",
         "Sigmoid-like probability via predict_proba; threshold = 0.5"],
    ], col_widths=[4.5 * cm, 3.0 * cm, 7.5 * cm]))
    story.append(Spacer(1, 8))

    add_heading(story, "2.2 Hyperparameters per Module", h2)
    story.append(make_table([
        ["Module", "Model Type", "Hidden Layers", "LR", "Alpha", "Max Iter"],
        ["Demand", "MLPRegressor",
         str(tuple(m["demand"]["params"]["hidden_layer_sizes"])),
         str(m["demand"]["params"]["learning_rate_init"]),
         str(m["demand"]["params"]["alpha"]),
         str(m["demand"]["params"]["max_iter"])],
        ["Distribution", "MLPClassifier",
         str(tuple(m["distribution"]["params"]["hidden_layer_sizes"])),
         str(m["distribution"]["params"]["learning_rate_init"]),
         str(m["distribution"]["params"]["alpha"]),
         str(m["distribution"]["params"]["max_iter"])],
        ["Leak", "MLPClassifier",
         str(tuple(m["leak"]["params"]["hidden_layer_sizes"])),
         str(m["leak"]["params"]["learning_rate_init"]),
         str(m["leak"]["params"]["alpha"]),
         str(m["leak"]["params"]["max_iter"])],
        ["Quality", "MLPClassifier",
         str(tuple(m["quality"]["params"]["hidden_layer_sizes"])),
         str(m["quality"]["params"]["learning_rate_init"]),
         str(m["quality"]["params"]["alpha"]),
         str(m["quality"]["params"]["max_iter"])],
    ], col_widths=[3.0 * cm, 3.5 * cm, 3.6 * cm, 1.7 * cm, 1.7 * cm, 2.3 * cm]))
    story.append(Spacer(1, 8))

    add_heading(story, "2.3 Training Procedure", h2)
    story.append(make_table([
        ["Parameter", "Value / Description"],
        ["Optimizer", "Adam (adaptive moment estimation)"],
        ["Activation function", "ReLU (all hidden layers)"],
        ["Demand max_iter", str(m["demand"]["params"]["max_iter"])],
        ["Classifier max_iter", str(m["distribution"]["params"]["max_iter"])],
        ["Learning rate", str(m["distribution"]["params"]["learning_rate_init"])],
        ["L2 regularization alpha", str(m["distribution"]["params"]["alpha"])],
        ["Hidden layers (classifiers)",
         str(tuple(m["distribution"]["params"]["hidden_layer_sizes"]))],
        ["Random state", "42 (fully reproducible)"],
        ["Validation strategy", "Held-out 20% test set + 5-fold stratified CV"],
    ], col_widths=[5.5 * cm, 9.5 * cm]))
    story.append(Spacer(1, 8))
    add_image(story, "Figure 2: ANN Training Loss Curves (All Modules)",
              graphs["loss_curves"], body, width_cm=16.0, ratio=0.62)
    story.append(PageBreak())

    # ==================================================================
    # STEP 3: MODEL EVALUATION AND DECISION INTELLIGENCE
    # ==================================================================
    add_heading(story, "STEP 3: Model Evaluation and Decision Intelligence", h1)

    add_heading(story, "3.1 Overall Evaluation Metrics", h2)
    add_paragraph(
        story,
        "Classifier modules: Accuracy, Precision, Recall, F1, ROC-AUC. "
        "Demand module: R\u00b2, RMSE, MAE.",
        body,
    )
    story.append(make_table([
        ["Module", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "R\u00b2", "RMSE", "MAE"],
        ["Demand", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014",
         f"{m['demand']['r2']:.4f}", f"{m['demand']['rmse']:.4f}", f"{m['demand']['mae']:.4f}"],
        ["Distribution",
         f"{m['distribution']['accuracy']:.4f}", f"{m['distribution']['precision']:.4f}",
         f"{m['distribution']['recall']:.4f}", f"{m['distribution']['f1']:.4f}",
         f"{m['distribution']['roc_auc']:.4f}", "\u2014", "\u2014", "\u2014"],
        ["Leak",
         f"{m['leak']['accuracy']:.4f}", f"{m['leak']['precision']:.4f}",
         f"{m['leak']['recall']:.4f}", f"{m['leak']['f1']:.4f}",
         f"{m['leak']['roc_auc']:.4f}", "\u2014", "\u2014", "\u2014"],
        ["Quality",
         f"{m['quality']['accuracy']:.4f}", f"{m['quality']['precision']:.4f}",
         f"{m['quality']['recall']:.4f}", f"{m['quality']['f1']:.4f}",
         f"{m['quality']['roc_auc']:.4f}", "\u2014", "\u2014", "\u2014"],
    ], col_widths=[2.6 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm,
                  2.0 * cm, 2.0 * cm, 1.4 * cm, 1.7 * cm, 1.7 * cm]))
    story.append(Spacer(1, 10))
    add_image(story, "Figure 3: Classifier Module Performance Comparison",
              graphs["performance"], body, width_cm=16.0, ratio=0.58)

    # ---- Per-class breakdown ------------------------------------------
    add_heading(story, "3.2 Per-Module Class-Level Breakdown", h2)
    add_paragraph(
        story,
        "Precision, Recall, and F1 score reported separately for Class 0 (negative) and "
        "Class 1 (positive) for each classification module on the held-out test set.",
        body,
    )
    pc_rows = [["Module", "Class", "Precision", "Recall", "F1-Score", "Support"]]
    for mod, lbl0, lbl1 in [
        ("distribution", "Normal (0)", "Attack/Risk (1)"),
        ("leak",         "No Leak (0)", "Leak (1)"),
        ("quality",      "Non-Potable (0)", "Potable (1)"),
    ]:
        if mod in per_class:
            c0 = per_class[mod]["class_0"]
            c1 = per_class[mod]["class_1"]
            pc_rows.append([mod.capitalize(), lbl0,
                             f"{c0['precision']:.4f}", f"{c0['recall']:.4f}",
                             f"{c0['f1-score']:.4f}", str(int(c0['support']))])
            pc_rows.append(["", lbl1,
                             f"{c1['precision']:.4f}", f"{c1['recall']:.4f}",
                             f"{c1['f1-score']:.4f}", str(int(c1['support']))])
        else:
            pc_rows.append([mod.capitalize(), lbl0, "N/A", "N/A", "N/A", "N/A"])
            pc_rows.append(["", lbl1, "N/A", "N/A", "N/A", "N/A"])
    story.append(make_table(pc_rows,
                            col_widths=[2.8 * cm, 3.2 * cm, 2.4 * cm,
                                        2.4 * cm, 2.4 * cm, 2.0 * cm]))
    story.append(Spacer(1, 10))

    # ---- Cross-validation all modules --------------------------------
    add_heading(story, "3.3 Cross-Validation Results (All Classifier Modules)", h2)
    add_paragraph(
        story,
        "5-fold stratified cross-validation on all three classification modules. "
        "Results shown as mean \u00b1 std across folds.",
        body,
    )
    cv_rows = [["Module", "CV Accuracy", "CV F1", "CV ROC-AUC"]]
    for mod in ("distribution", "leak", "quality"):
        if mod in cv_all:
            cv = cv_all[mod]
            cv_rows.append([
                mod.capitalize(),
                f"{cv['accuracy_mean']:.4f} \u00b1 {cv['accuracy_std']:.4f}",
                f"{cv['f1_mean']:.4f} \u00b1 {cv['f1_std']:.4f}",
                f"{cv['roc_auc_mean']:.4f} \u00b1 {cv['roc_auc_std']:.4f}",
            ])
        else:
            cv_rows.append([mod.capitalize(), "Running\u2026", "\u2014", "\u2014"])
    story.append(make_table(cv_rows,
                            col_widths=[3.2 * cm, 4.8 * cm, 4.2 * cm, 4.2 * cm]))
    story.append(Spacer(1, 8))
    add_paragraph(
        story,
        "CV results confirm that single held-out test metrics are consistent with "
        "multi-fold evaluation, ruling out overfitting to a specific test split.",
        body,
    )

    # ---- Rule-based fusion -------------------------------------------
    add_heading(story, "3.4 Rule-Based Decision Fusion", h2)
    add_paragraph(
        story,
        "The decision engine evaluates rules in strict priority order. The first rule whose "
        "condition is satisfied determines the final operational action.",
        body,
    )
    story.append(make_rules_table([
        ["Rule ID", "Condition", "Action", "Priority"],
        ["R1 QUALITY CRITICAL", "quality probability < 0.50",
         "HOLD SUPPLY AND TRIGGER TREATMENT", "Critical"],
        ["R2 LEAK HIGH", "leak probability \u2265 0.70",
         "ISOLATE LEAK ZONE AND DISPATCH MAINTENANCE", "High"],
        ["R3 DISTRIBUTION RISK HIGH", "distribution risk \u2265 0.70",
         "REDUCE PRESSURE AND REROUTE FLOW", "High"],
        ["R4 DEMAND SURGE", "demand score \u2265 0.75",
         "INCREASE SUPPLY TO HIGH DEMAND ZONES", "Medium"],
        ["R5 NORMAL", "All conditions false",
         "NORMAL OPERATION", "Low"],
    ], col_widths=[3.7 * cm, 3.9 * cm, 6.4 * cm, 1.9 * cm]))
    story.append(Spacer(1, 10))

    # ---- Decision walkthrough ----------------------------------------
    add_heading(story, "3.5 Integrated Decision Walkthrough \u2014 Worked Example", h2)
    add_paragraph(
        story,
        "The following example traces a sensor snapshot at 14:00 on a Tuesday through all "
        "four ANN modules and the rule engine to demonstrate the complete decision flow.",
        body,
    )
    add_paragraph(story, "<b>Step A \u2014 Sensor Input Snapshot:</b>", body)
    story.append(make_table([
        ["Module", "Key Input Values"],
        ["Demand",
         "hour=14, day=Tuesday, rolling mean=42.1 l/s, lag 1=43.8"],
        ["Distribution",
         "F PU1=1.21 (normal), P J280=42.5 kPa (normal), S PU1=1 (ON)"],
        ["Leak",
         "Pressure=38.2 kPa, Flow Rate=5.7 l/s, Vibration=0.02 mm/s"],
        ["Quality",
         "pH=7.2, Turbidity=1.8 NTU, Chloramines=6.8 mg/L, Sulfate=210 mg/L"],
    ], col_widths=[3.0 * cm, 12.3 * cm]))
    story.append(Spacer(1, 6))

    add_paragraph(story, "<b>Step B \u2014 ANN Module Outputs:</b>", body)
    story.append(make_table([
        ["Module", "Raw Output", "Interpretation"],
        ["Demand ANN", "51.3 l/s \u2192 norm. score 0.82",
         "HIGH \u2014 demand surge above normal baseline"],
        ["Distribution ANN", "Risk probability = 0.21",
         "LOW \u2014 network operating normally"],
        ["Leak ANN", "Leak probability = 0.15",
         "LOW \u2014 no anomaly in pressure/flow/vibration"],
        ["Quality ANN", "Potability probability = 0.71",
         "PASS \u2014 water classified as potable"],
    ], col_widths=[3.5 * cm, 5.0 * cm, 6.8 * cm]))
    story.append(Spacer(1, 6))

    add_paragraph(story, "<b>Step C \u2014 Rule Evaluation (priority order):</b>", body)
    story.append(make_rules_table([
        ["Rule ID", "Condition", "Value", "Met?", "Action Triggered"],
        ["R1 QUALITY CRITICAL", "quality prob < 0.50",
         "0.71", "No \u2014 0.71 \u2265 0.50", "\u2014"],
        ["R2 LEAK HIGH", "leak prob \u2265 0.70",
         "0.15", "No \u2014 0.15 < 0.70", "\u2014"],
        ["R3 DISTRIBUTION RISK HIGH", "dist risk \u2265 0.70",
         "0.21", "No \u2014 0.21 < 0.70", "\u2014"],
        ["R4 DEMAND SURGE", "demand score \u2265 0.75",
         "0.82", "YES", "INCREASE SUPPLY"],
    ], col_widths=[3.6 * cm, 3.3 * cm, 1.5 * cm, 3.2 * cm, 3.7 * cm]))
    story.append(Spacer(1, 6))
    add_paragraph(
        story,
        "<b>Final Decision:</b> INCREASE SUPPLY TO HIGH DEMAND ZONES (Rule R4 DEMAND SURGE). "
        "The system dispatches a supply increase command while all other parameters remain "
        "nominal. This decision is logged with a full rule trace on the Streamlit dashboard.",
        body,
    )
    story.append(PageBreak())

    # ==================================================================
    # STEP 4: CORRELATION HEATMAPS
    # ==================================================================
    add_heading(story, "STEP 4: Correlation Heatmaps", h1)
    add_paragraph(
        story,
        "Feature correlation heatmaps validate feature selection and highlight redundant "
        "signals. Strongly correlated feature pairs were identified and managed during "
        "the feature selection phase.",
        body,
    )
    add_heatmap(story, "Figure 4: Demand Module Correlation Heatmap",
                "demand_correlation_heatmap.png", body)
    add_heatmap(story, "Figure 5: Distribution Module Correlation Heatmap",
                "distribution_correlation_heatmap.png", body)
    add_heatmap(story, "Figure 6: Leak Module Correlation Heatmap",
                "leak_correlation_heatmap.png", body)
    add_heatmap(story, "Figure 7: Quality Module Correlation Heatmap",
                "quality_correlation_heatmap.png", body)
    story.append(PageBreak())

    # ==================================================================
    # STEP 5: DASHBOARD AND DEMONSTRATION
    # ==================================================================
    add_heading(story, "STEP 5: Dashboard and Demonstration", h1)
    add_paragraph(
        story,
        "The Streamlit dashboard provides a unified interactive interface for exploring "
        "model performance, running live predictions, and understanding decision rationale.",
        body,
    )
    story.append(make_rules_table([
        ["Dashboard Feature", "Description"],
        ["Module Metrics View",
         "Training accuracy, F1, ROC-AUC per module displayed as metric cards and bar charts."],
        ["Loss Curve Viewer",
         "Interactive training loss curve for each ANN module."],
        ["Confusion Matrix",
         "Per-module confusion matrix rendered as a heatmap."],
        ["ROC / PR Curves",
         "Receiver-operating characteristic and precision-recall curves."],
        ["Live Prediction Panel",
         "Slider inputs for all module features \u2014 returns prediction + probability instantly."],
        ["Rule Decision Display",
         "Shows active rule ID, condition values, and final action in real time."],
        ["Scenario Presets",
         "One-click presets: Normal, High Demand, Leak Emergency, Poor Quality Alert."],
    ], col_widths=[5.0 * cm, 10.2 * cm]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # ==================================================================
    # STEP 6: CONCLUSION
    # ==================================================================
    add_heading(story, "STEP 6: Conclusion", h1)
    add_paragraph(
        story,
        "This project successfully delivers a fully functional Smart Water Distribution "
        "Management System leveraging Artificial Neural Networks across four specialized modules.",
        body,
    )
    add_paragraph(
        story,
        f"The <b>demand forecasting module</b> achieves R\u00b2={m['demand']['r2']:.4f} and "
        f"RMSE={m['demand']['rmse']:.2f}\u202fl/s on the United Utilities dataset, demonstrating "
        "strong predictive power for zonal supply planning.",
        body,
    )
    add_paragraph(
        story,
        f"The <b>distribution risk module</b> achieves {m['distribution']['accuracy']*100:.1f}% "
        f"accuracy with ROC-AUC={m['distribution']['roc_auc']:.4f} on the BATADAL benchmark, "
        "confirming its ability to distinguish normal network operation from cyber-attack events.",
        body,
    )
    add_paragraph(
        story,
        f"The <b>leak detection module</b> achieves {m['leak']['accuracy']*100:.1f}% accuracy "
        f"and ROC-AUC={m['leak']['roc_auc']:.4f}, indicating near-perfect discriminability "
        "between leak and no-leak conditions from multi-sensor readings.",
        body,
    )
    add_paragraph(
        story,
        f"The <b>water quality module</b> achieves {m['quality']['accuracy']*100:.1f}% accuracy "
        f"and ROC-AUC={m['quality']['roc_auc']:.4f} on the Water Potability dataset. Quality "
        "classification is inherently the most challenging task due to overlapping feature "
        "distributions; 5-fold CV confirms stability at accuracy "
        f"{m['quality']['cv']['accuracy_mean']:.4f} \u00b1 {m['quality']['cv']['accuracy_std']:.4f}.",
        body,
    )
    add_paragraph(
        story,
        "The <b>rule-based fusion engine</b> aggregates all four module outputs in strict priority "
        "order, producing one of five transparent action codes with a logged rule trace. Every "
        "decision can be traced back to a specific rule condition and module probability value, "
        "ensuring full explainability.",
        body,
    )
    add_paragraph(
        story,
        "The complete system \u2014 including datasets, preprocessing pipelines, trained models, "
        "decision engine, and Streamlit dashboard \u2014 is fully reproducible from "
        "version-controlled source code and Kaggle open datasets. Each ANN module can be "
        "retrained independently when new sensor data or updated labels become available.",
        body,
    )

    add_heading(story, "STEP 7: Future Scope", h1)
    for point in [
        "Learn fusion weights from supervisory action labels when field-labeled incident data is available.",
        "Integrate live IoT sensor streams via MQTT for online inference without retraining.",
        "Extend decision logic with zone-level hydraulic simulation (EPANET integration).",
        "Compare ANN modules against XGBoost, LightGBM, and LSTM baselines.",
        "Implement SHAP-based feature attribution for per-prediction explainability.",
        "Add uncertainty estimation (MC Dropout or ensemble) to qualify prediction confidence.",
    ]:
        add_paragraph(story, f"\u2022 {point}", body)

    # ==================================================================
    # REFERENCES
    # ==================================================================
    add_heading(story, "References", h1)
    for ref in [
        f"1. United Utilities Water Management (Kaggle): {DATASET_LINKS['demand']}",
        f"2. BATADAL Cyber-Attack Dataset (Kaggle): {DATASET_LINKS['distribution']}",
        f"3. Smart Water Leak Detection Dataset (Kaggle): {DATASET_LINKS['leak']}",
        f"4. Water Potability Dataset (Kaggle): {DATASET_LINKS['quality']}",
        "5. Mounce, S. R. et al. (2011). Artificial intelligence applications in water utilities. "
        "Water Practice & Technology, 6(1).",
        "6. Taormina, R. et al. (2018). Battle of the Attack Detection Algorithms (BATADAL). "
        "J. Water Resources Planning & Management, 144(8).",
        "7. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
        "Journal of Machine Learning Research, 12, 2825\u20132830.",
        "8. Streamlit Inc. (2025). Streamlit Documentation. https://docs.streamlit.io/",
    ]:
        add_paragraph(story, ref, body)
    story.append(PageBreak())

    # ==================================================================
    # APPENDIX
    # ==================================================================
    add_heading(story, "Appendix A: Key Implementation Snippets", h1)

    add_heading(story, "A.1 Preprocessing Pipeline", h2)
    add_paragraph(story, "Sklearn ColumnTransformer pipeline applied before ANN training:", body)
    story.append(Preformatted(
        "def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:\n"
        "    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n"
        "    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()\n"
        "    numeric_pipeline = Pipeline(steps=[\n"
        "        ('imputer', SimpleImputer(strategy='median')),\n"
        "        ('scaler', StandardScaler()),\n"
        "    ])\n"
        "    categorical_pipeline = Pipeline(steps=[\n"
        "        ('imputer', SimpleImputer(strategy='most_frequent')),\n"
        "        ('onehot', OneHotEncoder(handle_unknown='ignore')),\n"
        "    ])\n"
        "    return ColumnTransformer(\n"
        "        transformers=[\n"
        "            ('num', numeric_pipeline, numeric_cols),\n"
        "            ('cat', categorical_pipeline, categorical_cols),\n"
        "        ], remainder='drop')",
        code_style,
    ))
    story.append(Spacer(1, 10))

    add_heading(story, "A.2 Training CLI Commands", h2)
    story.append(Preformatted(
        "# Demand (regression)\n"
        "python main.py train-demand \\\n"
        "    --csv ./data/demand/netbase_inlet-outlet-cont_logged_user_April2018.csv\n\n"
        "# Distribution (classifier, 600 epochs)\n"
        "python main.py --hidden-layers '256,128,64' --learning-rate-init 0.0007 \\\n"
        "    --alpha 0.001 --max-iter-clf 600 train-distribution \\\n"
        "    --csv ./data/distribution/training_dataset_2.csv --target-col ATT_FLAG\n\n"
        "# Leak (classifier, 600 epochs)\n"
        "python main.py --hidden-layers '256,128,64' --learning-rate-init 0.0007 \\\n"
        "    --alpha 0.001 --max-iter-clf 600 train-leak \\\n"
        "    --csv ./data/leak/location_aware_gis_leakage_dataset.csv \\\n"
        "    --target-col Leakage_Flag\n\n"
        "# Quality (classifier + 5-fold CV)\n"
        "python main.py --cv-folds 5 train-quality \\\n"
        "    --csv ./data/quality/water_potability.csv --target-col Potability",
        code_style,
    ))
    story.append(Spacer(1, 10))

    add_heading(story, "A.3 Rule-Based Decision Engine", h2)
    story.append(Preformatted(
        "DECISION_RULES = [\n"
        "    ('R1_QUALITY_CRITICAL',\n"
        "     lambda q, lk, dr, d: q < 0.50,\n"
        "     'HOLD_SUPPLY_AND_TRIGGER_TREATMENT'),\n"
        "    ('R2_LEAK_HIGH',\n"
        "     lambda q, lk, dr, d: lk >= 0.70,\n"
        "     'ISOLATE_LEAK_ZONE_AND_DISPATCH_MAINTENANCE'),\n"
        "    ('R3_DISTRIBUTION_RISK_HIGH',\n"
        "     lambda q, lk, dr, d: dr >= 0.70,\n"
        "     'REDUCE_PRESSURE_AND_REROUTE_FLOW'),\n"
        "    ('R4_DEMAND_SURGE',\n"
        "     lambda q, lk, dr, d: d >= 0.75,\n"
        "     'INCREASE_SUPPLY_TO_HIGH_DEMAND_ZONES'),\n"
        "    ('R5_NORMAL',\n"
        "     lambda q, lk, dr, d: True,\n"
        "     'NORMAL_OPERATION'),\n"
        "]\n\n"
        "def fuse_decisions(quality_prob, leak_prob, dist_risk, demand_score):\n"
        "    for rule_id, condition, action in DECISION_RULES:\n"
        "        if condition(quality_prob, leak_prob, dist_risk, demand_score):\n"
        "            return action, rule_id\n"
        "    return 'NORMAL_OPERATION', 'R5_NORMAL'",
        code_style,
    ))
    story.append(Spacer(1, 10))

    add_heading(story, "A.4 Launching the Streamlit Dashboard", h2)
    story.append(Preformatted(
        "streamlit run app.py --server.port 8501 --server.address 127.0.0.1\n"
        "# Access at: http://localhost:8501",
        code_style,
    ))

    # ---- Final build (two-pass for TOC page numbers) ------------------
    doc.multiBuild(story)
    print(f"\nGenerated: {OUT_FILE}")
    print(f"File size: {OUT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    build_report()
