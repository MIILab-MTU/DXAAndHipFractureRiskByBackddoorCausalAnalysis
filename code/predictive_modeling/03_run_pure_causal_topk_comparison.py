"""
Run pure causal Top-K feature comparison for hip fracture prediction.

This script is the cleaned GitHub-ready version of the previous predictive
modeling test3.py workflow.

It evaluates predictive performance using only the top-ranked causal DXA-derived
features. For K = 1, ..., max K, the model uses the first K causal features and
evaluates multiple classifiers with stratified cross-validation.

Main outputs:
    1. Fold-level Top-K results
    2. Mean ± SD summary table by TopK and model
    3. Best Top-K setting overall based on mean test AUC
    4. Best Top-K setting for each model
    5. Line plots for mean test AUC, PR-AUC, and F1 across TopK

Important:
    Patient-level data should NOT be uploaded to GitHub.
    Place the approved local dataset in a local data directory, or pass the path
    using --data-path.

Example:
    python scripts/predictive_modeling/03_run_pure_causal_topk_comparison.py \
        --data-path data/predictivemodel/all_features.csv \
        --out-dir results/predictive_modeling/pure_causal_topk

Optional:
    python scripts/predictive_modeling/03_run_pure_causal_topk_comparison.py \
        --data-path data/predictivemodel/all_features.csv \
        --out-dir results/predictive_modeling/pure_causal_topk \
        --max-k 16 \
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
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")


# =============================================================================
# Default settings
# =============================================================================

DEFAULT_TARGET_COL = "hip_fracture"

DEFAULT_TOP_CAUSAL_FEATURES = [
    "Femur_total_BMC(mean)",
    "Femur_total_BMD(mean)",
    "Femur_troch_BMD(mean)",
    "Femur_neck_BMD(mean)",
    "Femur_shaft_BMD(mean)",
    "Femur_total_BMD_T-score(mean)",
    "Femur_neck_BMD_T-score(mean)",
    "Femur_shaft_BMC(mean)",
    "Femur_wards_BMD(mean)",
    "Femur_troch_BMC(mean)",
    "Femur_wards_BMD_T-score(mean)",
    "Femur_troch_BMD_T-score(mean)",
    "Femur_neck_BMC(mean)",
    "Pelvis_BMD",
    "Pelvis_BMC",
    "Femur_wards_BMC(mean)",
]


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pure causal Top-K predictive modeling comparison."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data") / "predictivemodel" / "all_features.csv",
        help="Path to the local dataset containing all candidate causal features.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "predictive_modeling" / "pure_causal_topk",
        help="Directory for output tables and figures.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help="Binary outcome column name.",
    )
    parser.add_argument(
        "--top-causal-features",
        type=str,
        default="",
        help=(
            "Optional comma-separated ordered causal feature list. "
            "If empty, the default ordered list is used."
        ),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Maximum K to evaluate. If omitted, all available causal features are used.",
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
# Feature and data loading
# =============================================================================

def parse_top_causal_features(features_arg: str) -> list[str]:
    if not features_arg.strip():
        return DEFAULT_TOP_CAUSAL_FEATURES
    return [item.strip() for item in features_arg.split(",") if item.strip()]


def load_dataset(
    path: Path,
    target_col: str,
    top_causal_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Please place the approved local dataset in the data directory, "
            "or provide the correct --data-path."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {path.name}")

    available_causal = [col for col in top_causal_features if col in df.columns]
    missing_causal = [col for col in top_causal_features if col not in df.columns]

    if len(available_causal) == 0:
        raise ValueError("No causal features from the ordered list were found in the dataset.")

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid_target = y.notnull().values

    df = df.loc[valid_target].copy()
    y = y.loc[valid_target].astype(int).to_numpy()

    keep = np.isin(y, [0, 1])
    df = df.loc[keep].copy()
    y = y[keep]

    return df, y, available_causal, missing_causal


def validate_binary_cv_target(y: np.ndarray, n_splits: int) -> None:
    unique_values, counts = np.unique(y, return_counts=True)

    if len(unique_values) < 2:
        raise ValueError("Only one class found in the target variable.")

    min_class_count = int(np.min(counts))
    if min_class_count < n_splits:
        raise ValueError(
            f"The smallest class has only {min_class_count} samples, "
            f"which is smaller than n_splits={n_splits}."
        )


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


# =============================================================================
# Summary functions
# =============================================================================

def build_mean_std_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "Train_AUC",
        "Train_PRAUC",
        "Train_bestF1_Accuracy",
        "Train_bestF1_Sensitivity",
        "Train_bestF1_Specificity",
        "Train_bestF1_Precision",
        "Train_bestF1_Recall",
        "Train_bestF1_F1",
        "Test_AUC",
        "Test_PRAUC",
        "Test_bestF1_Accuracy",
        "Test_bestF1_Sensitivity",
        "Test_bestF1_Specificity",
        "Test_bestF1_Precision",
        "Test_bestF1_Recall",
        "Test_bestF1_F1",
        "BestThreshold_F1_fromTrain",
    ]

    value_columns = [col for col in value_columns if col in fold_df.columns]

    summary = (
        fold_df.groupby(["TopK", "Model"])[value_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()

    meta = (
        fold_df.groupby(["TopK", "Model"], as_index=False)
        .agg(
            n_total=("n_total", "first"),
            n_pos_total=("n_pos_total", "first"),
            n_features_after_dummy=("n_features_after_dummy", "first"),
            selected_causal_features=("SelectedCausalFeatures", "first"),
        )
    )

    summary = meta.merge(summary, on=["TopK", "Model"], how="left")
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


def select_best_overall(summary_df: pd.DataFrame, metric_col: str = "Test_AUC_mean") -> pd.DataFrame:
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()

    return summary_df.sort_values(metric_col, ascending=False).head(1).reset_index(drop=True)


def select_best_per_model(summary_df: pd.DataFrame, metric_col: str = "Test_AUC_mean") -> pd.DataFrame:
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()

    sorted_df = summary_df.sort_values(
        ["Model", metric_col],
        ascending=[True, False],
    ).copy()

    return sorted_df.groupby("Model", as_index=False).head(1).reset_index(drop=True)


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


def plot_metric_by_topk(
    summary_df: pd.DataFrame,
    metric_mean_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    save_pdf: bool,
) -> None:
    if summary_df.empty or metric_mean_col not in summary_df.columns:
        print(f"[WARN] Cannot create plot because {metric_mean_col} is missing.")
        return

    plot_df = summary_df.dropna(subset=[metric_mean_col]).copy()
    if plot_df.empty:
        print(f"[WARN] Cannot create plot because {metric_mean_col} has no valid values.")
        return

    plt.figure(figsize=(9, 5.5))

    for model_name, group in plot_df.groupby("Model"):
        group = group.sort_values("TopK")
        plt.plot(
            group["TopK"],
            group[metric_mean_col],
            marker="o",
            linewidth=1.5,
            label=model_name,
        )

    plt.xlabel("Top-K causal features")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(plot_df["TopK"].unique()))
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    save_current_figure(out_path, save_pdf=save_pdf)


def plot_best_model_by_topk(
    summary_df: pd.DataFrame,
    out_path: Path,
    save_pdf: bool,
) -> None:
    if summary_df.empty or "Test_AUC_mean" not in summary_df.columns:
        print("[WARN] Cannot create best-model-by-TopK plot because Test_AUC_mean is missing.")
        return

    best_by_k = (
        summary_df.sort_values(["TopK", "Test_AUC_mean"], ascending=[True, False])
        .groupby("TopK", as_index=False)
        .head(1)
        .sort_values("TopK")
        .copy()
    )

    if best_by_k.empty:
        return

    x = np.arange(len(best_by_k))
    labels = best_by_k["TopK"].astype(str).values
    values = best_by_k["Test_AUC_mean"].to_numpy(dtype=float)
    errors = (
        best_by_k["Test_AUC_std"].fillna(0).to_numpy(dtype=float)
        if "Test_AUC_std" in best_by_k.columns
        else None
    )

    plt.figure(figsize=(9, 5.5))
    plt.bar(x, values, yerr=errors, capsize=4)
    plt.xticks(x, labels)
    plt.xlabel("Top-K causal features")
    plt.ylabel("Mean test AUC")
    plt.title("Best Model Test AUC by Top-K Causal Feature Set")

    for idx, (_, row) in enumerate(best_by_k.iterrows()):
        plt.text(
            idx,
            values[idx],
            str(row["Model"]),
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

def prepare_feature_matrix(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    X = df[selected_features].copy()
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def run_cross_validation_for_topk(
    X: pd.DataFrame,
    y: np.ndarray,
    top_k: int,
    selected_causal_features: list[str],
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

            y_train_score = get_pred_scores(pipe, X_train)
            best_threshold = choose_best_threshold_by_f1(y_train, y_train_score)

            y_train_pred = (y_train_score >= best_threshold).astype(int)
            train_metrics = compute_metrics(y_train, y_train_score, y_train_pred)

            y_test_score = get_pred_scores(pipe, X_test)
            y_test_pred = (y_test_score >= best_threshold).astype(int)
            test_metrics = compute_metrics(y_test, y_test_score, y_test_pred)

            row = {
                "TopK": top_k,
                "Model": model_name,
                "Fold": fold_id,
                "n_total": len(y),
                "n_features_after_dummy": X.shape[1],
                "n_pos_total": int(np.sum(y == 1)),
                "n_pos_train": int(np.sum(y_train == 1)),
                "n_pos_test": int(np.sum(y_test == 1)),
                "BestThreshold_F1_fromTrain": best_threshold,
                "SelectedCausalFeatures": " | ".join(selected_causal_features),
            }

            row["Train_AUC"] = train_metrics["AUC"]
            row["Train_PRAUC"] = train_metrics["PRAUC"]
            row["Train_bestF1_Accuracy"] = train_metrics["Accuracy"]
            row["Train_bestF1_Sensitivity"] = train_metrics["Sensitivity"]
            row["Train_bestF1_Specificity"] = train_metrics["Specificity"]
            row["Train_bestF1_Precision"] = train_metrics["Precision"]
            row["Train_bestF1_Recall"] = train_metrics["Recall"]
            row["Train_bestF1_F1"] = train_metrics["F1"]

            row["Test_AUC"] = test_metrics["AUC"]
            row["Test_PRAUC"] = test_metrics["PRAUC"]
            row["Test_bestF1_Accuracy"] = test_metrics["Accuracy"]
            row["Test_bestF1_Sensitivity"] = test_metrics["Sensitivity"]
            row["Test_bestF1_Specificity"] = test_metrics["Specificity"]
            row["Test_bestF1_Precision"] = test_metrics["Precision"]
            row["Test_bestF1_Recall"] = test_metrics["Recall"]
            row["Test_bestF1_F1"] = test_metrics["F1"]

            rows.append(row)

    return rows


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    top_causal_features = parse_top_causal_features(args.top_causal_features)

    df, y, available_causal, missing_causal = load_dataset(
        path=args.data_path,
        target_col=args.target_col,
        top_causal_features=top_causal_features,
    )

    validate_binary_cv_target(y, args.n_splits)

    max_k = args.max_k if args.max_k is not None else len(available_causal)
    max_k = min(max_k, len(available_causal))

    if max_k < 1:
        raise ValueError("max_k must be at least 1.")

    if missing_causal:
        print("[WARN] The following causal features were not found and will be skipped:")
        for feature in missing_causal:
            print(f"  - {feature}")

    print(f"[INFO] Data path: {args.data_path}")
    print(f"[INFO] Output directory: {args.out_dir}")
    print(f"[INFO] Total samples: {len(y)}")
    print(f"[INFO] Positive samples: {int(np.sum(y == 1))}")
    print(f"[INFO] Available ordered causal features: {len(available_causal)}")
    print(f"[INFO] Maximum TopK evaluated: {max_k}")

    models = get_models(random_state=args.random_state)
    all_rows = []

    for top_k in range(1, max_k + 1):
        selected_features = available_causal[:top_k]
        X = prepare_feature_matrix(df, selected_features)

        print(
            f"\n[PURE CAUSAL TOP-K] K={top_k}, "
            f"selected raw features={len(selected_features)}, "
            f"features after encoding={X.shape[1]}"
        )

        rows = run_cross_validation_for_topk(
            X=X,
            y=y,
            top_k=top_k,
            selected_causal_features=selected_features,
            models=models,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        all_rows.extend(rows)

    fold_df = pd.DataFrame(all_rows)

    if fold_df.empty:
        print("[WARN] No fold-level results were generated.")
        return

    fold_path = args.out_dir / "pure_causal_topk_fold_results.csv"
    summary_path = args.out_dir / "pure_causal_topk_summary.csv"
    pretty_path = args.out_dir / "pure_causal_topk_summary_pretty.csv"
    best_path = args.out_dir / "pure_causal_topk_best_k.csv"
    best_per_model_path = args.out_dir / "pure_causal_topk_best_k_per_model.csv"

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved fold results: {fold_path}")

    summary_df = build_mean_std_summary(fold_df)
    summary_df = summary_df.sort_values(
        ["TopK", "Test_AUC_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved summary: {summary_path}")

    value_roots_for_pretty = [
        "Train_AUC",
        "Train_PRAUC",
        "Test_AUC",
        "Test_PRAUC",
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

    best_df = select_best_overall(summary_df, metric_col="Test_AUC_mean")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved best TopK overall: {best_path}")

    best_per_model_df = select_best_per_model(summary_df, metric_col="Test_AUC_mean")
    best_per_model_df.to_csv(best_per_model_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved best TopK per model: {best_per_model_path}")

    if not args.no_plots:
        save_pdf = not args.no_pdf

        plot_metric_by_topk(
            summary_df=summary_df,
            metric_mean_col="Test_AUC_mean",
            out_path=args.out_dir / "Figure_pure_causal_topk_AUC.png",
            title="Pure Causal Top-K Prediction: Test AUC",
            ylabel="Mean test AUC",
            save_pdf=save_pdf,
        )

        plot_metric_by_topk(
            summary_df=summary_df,
            metric_mean_col="Test_PRAUC_mean",
            out_path=args.out_dir / "Figure_pure_causal_topk_PRAUC.png",
            title="Pure Causal Top-K Prediction: Test PR-AUC",
            ylabel="Mean test PR-AUC",
            save_pdf=save_pdf,
        )

        plot_metric_by_topk(
            summary_df=summary_df,
            metric_mean_col="Test_bestF1_F1_mean",
            out_path=args.out_dir / "Figure_pure_causal_topk_F1.png",
            title="Pure Causal Top-K Prediction: Test F1",
            ylabel="Mean test F1",
            save_pdf=save_pdf,
        )

        plot_best_model_by_topk(
            summary_df=summary_df,
            out_path=args.out_dir / "Figure_pure_causal_topk_best_model_by_K.png",
            save_pdf=save_pdf,
        )

    print("[OK] Pure causal Top-K prediction workflow finished.")


if __name__ == "__main__":
    main()
