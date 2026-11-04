"""
Run predictive model comparison across multiple feature-set datasets.

This script is the cleaned GitHub-ready version of the previous predictive
modeling test4.py workflow.

It evaluates multiple machine-learning classifiers across prespecified
feature-set CSV files using 5-fold stratified cross-validation.

Main outputs:
    1. Fold-level train/test performance table
    2. Mean ± SD summary table by dataset and model
    3. Pretty summary table with formatted mean ± SD columns
    4. Best model per dataset based on mean test AUC
    5. Boxplots for test AUC and test PR-AUC
    6. Bar plot for best model test AUC by dataset

Important:
    Patient-level data should NOT be uploaded to GitHub.
    Place the approved local feature-set CSV files in a local data directory,
    or pass the path using --data-dir.

Example:
    python scripts/predictive_modeling/01_run_predictive_model_comparison.py \
        --data-dir data/predictivemodel \
        --out-dir results/predictive_modeling/model_comparison

Optional:
    python scripts/predictive_modeling/01_run_predictive_model_comparison.py \
        --data-dir data/predictivemodel \
        --out-dir results/predictive_modeling/model_comparison \
        --n-splits 5 \
        --random-state 2025
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings("ignore")


# =============================================================================
# Default settings
# =============================================================================

DEFAULT_TARGET_COL = "hip_fracture"

DEFAULT_DATASET_FILES = [
    "all_features.csv",
    "clinical_only.csv",
    "clinical_top2_causal.csv",
    "clinical_top3_causal.csv",
    "clinical_top4_causal.csv",
    "clinical_top5_causal.csv",
    "clinical_top6_causal.csv",
    "clinical_top7_causal.csv",
    "clinical_top8_causal.csv",
    "clinical_top9_causal.csv",
    "clinical_top10_causal.csv",
    "clinical_top11_causal.csv",
    "top2_causal_only.csv",
    "top3_causal_only.csv",
    "top4_causal_only.csv",
    "top5_causal_only.csv",
    "top6_causal_only.csv",
    "top7_causal_only.csv",
    "top8_causal_only.csv",
    "top9_causal_only.csv",
    "top10_causal_only.csv",
    "top11_causal_only.csv",
    "all_causal.csv",
]


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictive model comparison across feature-set datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "predictivemodel",
        help="Directory containing feature-set CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "predictive_modeling" / "model_comparison",
        help="Directory for output tables and figures.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help="Binary outcome column name.",
    )
    parser.add_argument(
        "--dataset-files",
        type=str,
        default="",
        help=(
            "Optional comma-separated dataset file names. "
            "If empty, the default feature-set list is used."
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2025,
        help="Random seed for cross-validation and applicable models.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Save tables only and skip figure generation.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Save PNG figures only and skip PDF output.",
    )
    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def parse_dataset_files(dataset_files_arg: str) -> list[str]:
    if not dataset_files_arg.strip():
        return DEFAULT_DATASET_FILES
    return [item.strip() for item in dataset_files_arg.split(",") if item.strip()]


def resolve_file_path(data_dir: Path, file_name: str) -> Path | None:
    candidates = [
        data_dir / file_name,
    ]

    if not str(file_name).lower().endswith(".csv"):
        candidates.append(data_dir / f"{file_name}.csv")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def load_dataset(path: Path, target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {path.name}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col]).copy()

    # One-hot encode categorical variables.
    X = pd.get_dummies(X, drop_first=True)

    # Force numeric values.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove missing or non-binary target rows.
    valid_target = y.notnull().values
    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].astype(int).to_numpy()

    keep = np.isin(y, [0, 1])
    X = X.loc[keep].copy()
    y = y[keep]

    return X, y


def validate_binary_cv_target(y: np.ndarray, n_splits: int, dataset_name: str) -> bool:
    unique_values, counts = np.unique(y, return_counts=True)

    if len(unique_values) < 2:
        print(f"[WARN] Only one class found in {dataset_name}. Skipping.")
        return False

    min_class_count = int(np.min(counts))
    if min_class_count < n_splits:
        print(
            f"[WARN] The smallest class in {dataset_name} has only {min_class_count} samples, "
            f"which is smaller than n_splits={n_splits}. Skipping."
        )
        return False

    return True


# =============================================================================
# Model definitions
# =============================================================================

def get_models(random_state: int = 2025) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            random_state=random_state,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=random_state,
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
        ),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=random_state,
        ),
    }


def make_pipeline(model_name: str, model: object) -> Pipeline:
    scale_models = {
        "LogisticRegression",
        "SVM_RBF",
        "KNN",
        "GaussianNB",
        "MLP",
    }

    steps = [("imputer", SimpleImputer(strategy="median"))]

    if model_name in scale_models:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))
    return Pipeline(steps)


# =============================================================================
# Metric functions
# =============================================================================

def get_pred_scores(fitted_pipeline: Pipeline, X_data: pd.DataFrame) -> np.ndarray:
    if hasattr(fitted_pipeline, "predict_proba"):
        scores = fitted_pipeline.predict_proba(X_data)[:, 1]
    elif hasattr(fitted_pipeline, "decision_function"):
        scores = fitted_pipeline.decision_function(X_data)
    else:
        scores = fitted_pipeline.predict(X_data).astype(float)

    return np.asarray(scores, dtype=float)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "AUC": np.nan,
        "PRAUC": np.nan,
        "Accuracy": np.nan,
        "F1": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "Sensitivity": np.nan,
        "Specificity": np.nan,
    }

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    try:
        if len(np.unique(y_true)) > 1:
            out["AUC"] = roc_auc_score(y_true, y_score)
            out["PRAUC"] = average_precision_score(y_true, y_score)
    except Exception:
        pass

    try:
        out["Accuracy"] = accuracy_score(y_true, y_pred)
        out["F1"] = f1_score(y_true, y_pred, zero_division=0)
        out["Precision"] = precision_score(y_true, y_pred, zero_division=0)
        out["Recall"] = recall_score(y_true, y_pred, zero_division=0)
        out["Sensitivity"] = out["Recall"]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        if (tn + fp) > 0:
            out["Specificity"] = tn / (tn + fp)
    except Exception:
        pass

    return out


def choose_best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Select a classification threshold on the training fold that maximizes F1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if len(thresholds) == 0:
        return 0.5

    precision_t = precision[:-1]
    recall_t = recall[:-1]

    denominator = precision_t + recall_t
    f1_values = np.where(
        denominator > 0,
        2 * precision_t * recall_t / denominator,
        0.0,
    )

    if len(f1_values) == 0:
        return 0.5

    best_index = int(np.nanargmax(f1_values))
    best_threshold = float(thresholds[best_index])

    if not np.isfinite(best_threshold):
        return 0.5

    return best_threshold


def add_metric_columns_to_row(
    row: dict,
    prefix: str,
    metrics: dict[str, float],
    predicted_positive_count: int,
) -> None:
    row[f"{prefix}_Accuracy"] = metrics["Accuracy"]
    row[f"{prefix}_Sensitivity"] = metrics["Sensitivity"]
    row[f"{prefix}_Specificity"] = metrics["Specificity"]
    row[f"{prefix}_Precision"] = metrics["Precision"]
    row[f"{prefix}_Recall"] = metrics["Recall"]
    row[f"{prefix}_F1"] = metrics["F1"]
    row[f"{prefix}_PredPos"] = predicted_positive_count


# =============================================================================
# Summary functions
# =============================================================================

def metric_value_columns() -> list[str]:
    return [
        "Train_AUC",
        "Train_PRAUC",
        "Train_default05_Accuracy",
        "Train_default05_Sensitivity",
        "Train_default05_Specificity",
        "Train_default05_Precision",
        "Train_default05_Recall",
        "Train_default05_F1",
        "Train_bestF1_Accuracy",
        "Train_bestF1_Sensitivity",
        "Train_bestF1_Specificity",
        "Train_bestF1_Precision",
        "Train_bestF1_Recall",
        "Train_bestF1_F1",
        "Test_AUC",
        "Test_PRAUC",
        "Test_default05_Accuracy",
        "Test_default05_Sensitivity",
        "Test_default05_Specificity",
        "Test_default05_Precision",
        "Test_default05_Recall",
        "Test_default05_F1",
        "Test_bestF1_Accuracy",
        "Test_bestF1_Sensitivity",
        "Test_bestF1_Specificity",
        "Test_bestF1_Precision",
        "Test_bestF1_Recall",
        "Test_bestF1_F1",
        "BestThreshold_F1_fromTrain",
        "Train_default05_PredPos",
        "Train_bestF1_PredPos",
        "Test_default05_PredPos",
        "Test_bestF1_PredPos",
    ]


def build_mean_std_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    value_columns = [col for col in metric_value_columns() if col in fold_df.columns]

    summary = (
        fold_df.groupby(["Dataset", "Model"])[value_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()

    meta = (
        fold_df.groupby(["Dataset", "Model"], as_index=False)
        .agg(
            n_features=("n_features", "first"),
            n_total=("n_total", "first"),
            n_pos_total=("n_pos_total", "first"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
        )
    )

    summary = meta.merge(summary, on=["Dataset", "Model"], how="left")
    return summary


def add_mean_pm_sd_columns(summary_df: pd.DataFrame, value_roots: list[str], digits: int = 4) -> pd.DataFrame:
    out = summary_df.copy()

    for root in value_roots:
        mean_col = f"{root}_mean"
        std_col = f"{root}_std"

        if mean_col in out.columns and std_col in out.columns:
            def format_value(row):
                if pd.notnull(row[mean_col]) and pd.notnull(row[std_col]):
                    return f"{row[mean_col]:.{digits}f} ± {row[std_col]:.{digits}f}"
                if pd.notnull(row[mean_col]):
                    return f"{row[mean_col]:.{digits}f} ± NA"
                return "NA"

            out[f"{root}_mean_sd"] = out.apply(format_value, axis=1)

    return out


def select_best_models(summary_df: pd.DataFrame, metric_col: str = "Test_AUC_mean") -> pd.DataFrame:
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()

    sorted_df = summary_df.sort_values(
        ["Dataset", metric_col],
        ascending=[True, False],
    ).copy()

    best = (
        sorted_df.groupby("Dataset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best


# =============================================================================
# Plot functions
# =============================================================================

def save_current_figure(out_path: Path, save_pdf: bool = True, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved PNG: {out_path}")

    if save_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"[OK] Saved PDF: {pdf_path}")

    plt.close()


def plot_metric_boxplot(
    fold_df: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    save_pdf: bool,
) -> None:
    if fold_df.empty or metric_col not in fold_df.columns:
        print(f"[WARN] Cannot create boxplot because {metric_col} is missing.")
        return

    plot_df = fold_df.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        print(f"[WARN] Cannot create boxplot because {metric_col} has no valid values.")
        return

    dataset_order = (
        plot_df.groupby("Dataset")[metric_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    data = [
        plot_df.loc[plot_df["Dataset"] == dataset, metric_col].values
        for dataset in dataset_order
    ]

    fig_width = max(10, 0.45 * len(dataset_order))
    plt.figure(figsize=(fig_width, 5.5))
    plt.boxplot(data, labels=dataset_order, showmeans=True)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    save_current_figure(out_path, save_pdf=save_pdf)


def plot_best_model_bar(
    best_df: pd.DataFrame,
    out_path: Path,
    save_pdf: bool,
) -> None:
    if best_df.empty or "Test_AUC_mean" not in best_df.columns:
        print("[WARN] Cannot create best-model bar plot because Test_AUC_mean is missing.")
        return

    plot_df = best_df.sort_values("Test_AUC_mean", ascending=False).copy()

    x = np.arange(len(plot_df))
    labels = plot_df["Dataset"].astype(str).values
    values = plot_df["Test_AUC_mean"].to_numpy(dtype=float)
    errors = (
        plot_df["Test_AUC_std"].fillna(0).to_numpy(dtype=float)
        if "Test_AUC_std" in plot_df.columns
        else None
    )

    fig_width = max(10, 0.5 * len(plot_df))
    plt.figure(figsize=(fig_width, 5.5))
    plt.bar(x, values, yerr=errors, capsize=4)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Mean test AUC")
    plt.xlabel("Dataset")
    plt.title("Best Model Test AUC by Dataset")

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        model_name = str(row["Model"])
        plt.text(
            idx,
            values[idx],
            model_name,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )

    plt.tight_layout()
    save_current_figure(out_path, save_pdf=save_pdf)


# =============================================================================
# Cross-validation workflow
# =============================================================================

def run_cross_validation_for_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    dataset_name: str,
    models: dict[str, object],
    n_splits: int,
    random_state: int,
) -> list[dict]:
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for model_name, model in models.items():
        print(f"  [MODEL] {model_name}")

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y[train_idx]
            y_test = y[test_idx]

            pipe = make_pipeline(model_name, model)
            pipe.fit(X_train, y_train)

            # Training fold scores and thresholds.
            y_train_score = get_pred_scores(pipe, X_train)
            best_threshold = choose_best_threshold_by_f1(y_train, y_train_score)

            y_train_pred_default05 = (y_train_score >= 0.5).astype(int)
            y_train_pred_bestF1 = (y_train_score >= best_threshold).astype(int)

            train_metrics_default05 = compute_metrics(
                y_train,
                y_train_score,
                y_train_pred_default05,
            )
            train_metrics_bestF1 = compute_metrics(
                y_train,
                y_train_score,
                y_train_pred_bestF1,
            )

            # Test fold scores.
            y_test_score = get_pred_scores(pipe, X_test)

            y_test_pred_default05 = (y_test_score >= 0.5).astype(int)
            y_test_pred_bestF1 = (y_test_score >= best_threshold).astype(int)

            test_metrics_default05 = compute_metrics(
                y_test,
                y_test_score,
                y_test_pred_default05,
            )
            test_metrics_bestF1 = compute_metrics(
                y_test,
                y_test_score,
                y_test_pred_bestF1,
            )

            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Fold": fold_id,
                "n_total": len(y),
                "n_features": X.shape[1],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_pos_total": int(np.sum(y == 1)),
                "n_pos_train": int(np.sum(y_train == 1)),
                "n_pos_test": int(np.sum(y_test == 1)),
                "BestThreshold_F1_fromTrain": best_threshold,
            }

            # Threshold-free metrics.
            row["Train_AUC"] = train_metrics_default05["AUC"]
            row["Train_PRAUC"] = train_metrics_default05["PRAUC"]
            row["Test_AUC"] = test_metrics_default05["AUC"]
            row["Test_PRAUC"] = test_metrics_default05["PRAUC"]

            # Metrics using default threshold 0.5.
            add_metric_columns_to_row(
                row,
                "Train_default05",
                train_metrics_default05,
                int(np.sum(y_train_pred_default05 == 1)),
            )
            add_metric_columns_to_row(
                row,
                "Test_default05",
                test_metrics_default05,
                int(np.sum(y_test_pred_default05 == 1)),
            )

            # Metrics using training-fold best-F1 threshold.
            add_metric_columns_to_row(
                row,
                "Train_bestF1",
                train_metrics_bestF1,
                int(np.sum(y_train_pred_bestF1 == 1)),
            )
            add_metric_columns_to_row(
                row,
                "Test_bestF1",
                test_metrics_bestF1,
                int(np.sum(y_test_pred_bestF1 == 1)),
            )

            rows.append(row)

    return rows


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = parse_dataset_files(args.dataset_files)
    models = get_models(random_state=args.random_state)

    print(f"[INFO] Data directory: {args.data_dir}")
    print(f"[INFO] Output directory: {args.out_dir}")
    print(f"[INFO] Number of dataset files requested: {len(dataset_files)}")
    print(f"[INFO] Number of models: {len(models)}")

    all_fold_rows = []

    for file_name in dataset_files:
        file_path = resolve_file_path(args.data_dir, file_name)

        if file_path is None:
            print(f"[WARN] File not found, skipped: {file_name}")
            continue

        dataset_name = file_path.stem
        print(f"\n[DATASET] {dataset_name}")

        X, y = load_dataset(file_path, args.target_col)
        print(
            f"[INFO] X shape = {X.shape}, "
            f"n = {len(y)}, positive = {int(np.sum(y == 1))}"
        )

        if not validate_binary_cv_target(y, args.n_splits, dataset_name):
            continue

        rows = run_cross_validation_for_dataset(
            X=X,
            y=y,
            dataset_name=dataset_name,
            models=models,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        all_fold_rows.extend(rows)

    fold_df = pd.DataFrame(all_fold_rows)

    if fold_df.empty:
        print("[WARN] No fold-level results were generated.")
        return

    fold_path = args.out_dir / "predictive_modeling_fold_results_train_test_with_thresholds.csv"
    summary_path = args.out_dir / "predictive_modeling_summary_train_test_with_thresholds.csv"
    pretty_path = args.out_dir / "predictive_modeling_summary_train_test_with_thresholds_pretty.csv"
    best_path = args.out_dir / "predictive_modeling_best_models_train_test_with_thresholds.csv"
    best_pretty_path = args.out_dir / "predictive_modeling_best_models_train_test_with_thresholds_pretty.csv"

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved fold results: {fold_path}")

    summary_df = build_mean_std_summary(fold_df)
    summary_df = summary_df.sort_values(
        ["Dataset", "Test_AUC_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved summary: {summary_path}")

    value_roots_for_pretty = [
        "Train_AUC",
        "Train_PRAUC",
        "Test_AUC",
        "Test_PRAUC",
        "Test_default05_Accuracy",
        "Test_default05_Sensitivity",
        "Test_default05_Specificity",
        "Test_default05_Precision",
        "Test_default05_F1",
        "Test_bestF1_Accuracy",
        "Test_bestF1_Sensitivity",
        "Test_bestF1_Specificity",
        "Test_bestF1_Precision",
        "Test_bestF1_F1",
        "BestThreshold_F1_fromTrain",
    ]

    pretty_df = add_mean_pm_sd_columns(summary_df, value_roots_for_pretty)
    pretty_df.to_csv(pretty_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved pretty summary: {pretty_path}")

    best_df = select_best_models(summary_df, metric_col="Test_AUC_mean")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved best models: {best_path}")

    best_pretty_df = add_mean_pm_sd_columns(best_df, value_roots_for_pretty)
    best_pretty_df.to_csv(best_pretty_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved pretty best models: {best_pretty_path}")

    if not args.no_plots:
        save_pdf = not args.no_pdf

        plot_metric_boxplot(
            fold_df=fold_df,
            metric_col="Test_AUC",
            out_path=args.out_dir / "Figure_TEST_AUC_boxplot_by_dataset.png",
            title="Test AUC by Feature-Set Dataset",
            ylabel="Test AUC",
            save_pdf=save_pdf,
        )

        plot_metric_boxplot(
            fold_df=fold_df,
            metric_col="Test_PRAUC",
            out_path=args.out_dir / "Figure_TEST_PRAUC_boxplot_by_dataset.png",
            title="Test PR-AUC by Feature-Set Dataset",
            ylabel="Test PR-AUC",
            save_pdf=save_pdf,
        )

        plot_best_model_bar(
            best_df=best_df,
            out_path=args.out_dir / "Figure_best_model_TEST_AUC_by_dataset.png",
            save_pdf=save_pdf,
        )

    print("[OK] Predictive model comparison workflow finished.")


if __name__ == "__main__":
    main()
