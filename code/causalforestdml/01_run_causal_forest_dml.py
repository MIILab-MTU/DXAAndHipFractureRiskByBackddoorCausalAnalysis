"""
Run CausalForestDML analysis for DXA-derived bone features and hip fracture.

This GitHub-ready script combines the all-treatment CausalForestDML workflow
and the CATE heterogeneity workflow from the earlier test3.py and test4.py files.

Important:
    Patient-level data and individual-level CATE outputs should NOT be uploaded
    to GitHub. Keep data/ and results/ ignored by Git.

Example:
    python code/causalforestdml/01_run_causal_forest_dml.py \
        --data-path data/BackdoorData.csv \
        --out-dir results/causalforestdml
"""

from __future__ import annotations

import argparse
import re
import traceback
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

OUTCOME_COL = "hip_fracture"
AGE_COL = "Age"
SEX_COL = "Sex"
HEIGHT_COL = "Height"
WEIGHT_COL = "Weight"
BMI_COL = "Body mass index (BMI)"

TREATMENT_COLS = [
    "Femur_neck_BMC(mean)",
    "Femur_neck_BMD(mean)",
    "Femur_neck_BMD_T-score(mean)",
    "Femur_total_BMC(mean)",
    "Femur_total_BMD(mean)",
    "Femur_total_BMD_T-score(mean)",
    "Femur_troch_BMC(mean)",
    "Femur_troch_BMD(mean)",
    "Femur_troch_BMD_T-score(mean)",
    "Femur_wards_BMC(mean)",
    "Femur_wards_BMD(mean)",
    "Femur_wards_BMD_T-score(mean)",
    "Femur_shaft_BMC(mean)",
    "Femur_shaft_BMD(mean)",
    "Pelvis_BMC",
    "Pelvis_BMD",
]

CONTROL_COLS = [
    "Age",
    "Sex",
    "Height",
    "Weight",
    "Household_income",
    "Smoking",
    "Rheumatoid_arthritis",
    "Dietary_changes_last5years",
    "Falls_lastyear",
    "Fractured/broken_bones_last5years",
    "Usual_walking_pace",
    "Hand_grip_stress(mean)",
    "Amount_of_alcohol_drunk_on_a_drinking_day",
]

HETEROGENEITY_COLS = CONTROL_COLS.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CausalForestDML analysis.")
    parser.add_argument("--data-path", type=Path, default=Path("data") / "BackdoorData.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "causal_forest")
    parser.add_argument("--complete-case", choices=["per-treatment", "all"], default="per-treatment")
    parser.add_argument("--random-state", type=int, default=2025)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--nuisance-trees", type=int, default=300)
    parser.add_argument("--cf-trees", type=int, default=1000)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--min-n", type=int, default=200)
    parser.add_argument("--min-unique-treatment", type=int, default=10)
    parser.add_argument("--topk-importance", type=int, default=12)
    parser.add_argument("--treatments", type=str, default="")
    parser.add_argument("--skip-bmi-recompute", action="store_true")
    parser.add_argument("--no-save-cate-predictions", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def safe_name(value: str, max_len: int = 180) -> str:
    value = str(value).strip()
    value = re.sub(r"[\\/*?:\"<>|]", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value[:max_len]


def deduplicate_columns_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


def build_unique_column_list(cols: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def get_first_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    obj = df.loc[:, col_name]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj)
    obj.name = col_name
    return obj


def validate_columns(df: pd.DataFrame, required_cols: Iterable[str], context: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for {context}:\n" + "\n".join(f"  - {col}" for col in missing)
        )


def maybe_stratify_target(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    values = np.unique(y[np.isfinite(y)])
    if not set(values).issubset({0.0, 1.0}) or len(values) < 2:
        return None
    y_int = y.astype(int)
    _, counts = np.unique(y_int, return_counts=True)
    return y_int if np.min(counts) >= 2 else None


def classify_family(treatment_name: str) -> str:
    x = str(treatment_name).lower()
    if "t-score" in x:
        return "T-score"
    if "bmd" in x:
        return "BMD"
    if "bmc" in x:
        return "BMC"
    return "Other"


def classify_region(treatment_name: str) -> str:
    x = str(treatment_name).lower()
    if "neck" in x:
        return "Neck"
    if "total" in x:
        return "Total"
    if "troch" in x:
        return "Troch"
    if "wards" in x:
        return "Wards"
    if "shaft" in x:
        return "Shaft"
    if "pelvis" in x:
        return "Pelvis"
    return "Other"


def short_treatment_label(treatment_name: str) -> str:
    region_label = {
        "Neck": "Femoral neck",
        "Total": "Total femur",
        "Troch": "Trochanter",
        "Wards": "Ward's region",
        "Shaft": "Femoral shaft",
        "Pelvis": "Pelvis",
    }.get(classify_region(treatment_name), "Other region")
    return f"{region_label} {classify_family(treatment_name)}"


def save_current_figure(out_path_png: Path, save_pdf: bool = True, dpi: int = 300) -> None:
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved PNG: {out_path_png}")
    if save_pdf:
        pdf_path = out_path_png.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"[OK] Saved PDF: {pdf_path}")
    plt.close()


def compute_or_update_bmi(df: pd.DataFrame, skip_recompute: bool = False) -> pd.DataFrame:
    if skip_recompute:
        return df
    if HEIGHT_COL not in df.columns or WEIGHT_COL not in df.columns:
        print("[WARN] Height or Weight not available. BMI cannot be computed.")
        return df
    height = pd.to_numeric(df[HEIGHT_COL], errors="coerce").mask(lambda s: s <= 0)
    weight = pd.to_numeric(df[WEIGHT_COL], errors="coerce").mask(lambda s: s <= 0)
    df[BMI_COL] = weight / ((height / 100.0) ** 2)
    print(f"[INFO] BMI column computed/updated: {BMI_COL}")
    return df


def select_treatments(treatments_arg: str) -> list[str]:
    if not treatments_arg.strip():
        return TREATMENT_COLS
    return [item.strip() for item in treatments_arg.split(",") if item.strip()]


def prepare_single_treatment_data(
    df: pd.DataFrame,
    treatment_col: str,
    x_cols: list[str],
    w_cols: list[str],
    all_treatment_cols: list[str],
    complete_case: str,
) -> dict:
    optional_cols = [BMI_COL] if BMI_COL in df.columns else []
    if complete_case == "all":
        need_cols = build_unique_column_list(
            [OUTCOME_COL] + all_treatment_cols + x_cols + w_cols + [AGE_COL, SEX_COL] + optional_cols
        )
    else:
        need_cols = build_unique_column_list([OUTCOME_COL, treatment_col, AGE_COL, SEX_COL] + optional_cols + x_cols + w_cols)

    validate_columns(df, need_cols, context=f"treatment {treatment_col}")
    df_use = deduplicate_columns_keep_first(df.loc[:, need_cols].copy())

    y_s = pd.to_numeric(get_first_series(df_use, OUTCOME_COL), errors="coerce")
    t_s = pd.to_numeric(get_first_series(df_use, treatment_col), errors="coerce")
    age_s = pd.to_numeric(get_first_series(df_use, AGE_COL), errors="coerce")
    sex_raw = get_first_series(df_use, SEX_COL).astype(str)
    bmi_s = (
        pd.to_numeric(get_first_series(df_use, BMI_COL), errors="coerce")
        if BMI_COL in df_use.columns
        else pd.Series(np.nan, index=df_use.index, name=BMI_COL)
    )

    X_df = pd.get_dummies(deduplicate_columns_keep_first(df_use.loc[:, x_cols].copy()), drop_first=True)
    W_df = pd.get_dummies(deduplicate_columns_keep_first(df_use.loc[:, w_cols].copy()), drop_first=True)
    X_df = deduplicate_columns_keep_first(X_df).apply(pd.to_numeric, errors="coerce")
    W_df = deduplicate_columns_keep_first(W_df).apply(pd.to_numeric, errors="coerce")

    valid_mask = (
        y_s.notnull().values
        & t_s.notnull().values
        & age_s.notnull().values
        & sex_raw.notnull().values
        & X_df.notnull().all(axis=1).values
        & W_df.notnull().all(axis=1).values
    )

    if complete_case == "all":
        cc_cols = [col for col in build_unique_column_list([OUTCOME_COL] + all_treatment_cols + x_cols + w_cols) if col in df_use.columns]
        valid_mask = valid_mask & df_use[cc_cols].notnull().all(axis=1).values

    y_s = y_s.loc[valid_mask].astype(float)
    t_s = t_s.loc[valid_mask].astype(float)
    age_s = age_s.loc[valid_mask].astype(float)
    sex_raw = sex_raw.loc[valid_mask].astype(str)
    bmi_s = bmi_s.loc[valid_mask]
    X_df = X_df.loc[valid_mask].dropna(axis=1, how="all").copy()
    W_df = W_df.loc[valid_mask].dropna(axis=1, how="all").copy()

    if X_df.shape[1] == 0 or W_df.shape[1] == 0:
        raise ValueError(f"X or W became empty after cleaning for {treatment_col}.")

    Y = y_s.to_numpy(dtype=float)
    T_raw = t_s.to_numpy(dtype=float)
    t_sd = float(np.std(T_raw, ddof=0))
    if not np.isfinite(t_sd) or t_sd == 0:
        raise ValueError(f"Invalid treatment SD for {treatment_col}: {t_sd}")

    return {
        "Y": Y,
        "T_raw": T_raw,
        "T_scaled": T_raw / t_sd,
        "X": X_df.to_numpy(dtype=float),
        "W": W_df.to_numpy(dtype=float),
        "X_df": X_df,
        "W_df": W_df,
        "age": age_s.to_numpy(dtype=float),
        "sex_raw": sex_raw.to_numpy(),
        "bmi": pd.to_numeric(bmi_s, errors="coerce").to_numpy(dtype=float),
        "t_sd": t_sd,
        "n_total": len(Y),
        "n_y1": int(np.sum(Y == 1)),
        "n_y0": int(np.sum(Y == 0)),
        "n_unique_t": int(len(np.unique(T_raw))),
    }


def fit_causal_forest(data: dict, args: argparse.Namespace, train_idx: np.ndarray) -> CausalForestDML:
    model_y = RandomForestRegressor(
        n_estimators=args.nuisance_trees,
        min_samples_leaf=10,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model_t = RandomForestRegressor(
        n_estimators=args.nuisance_trees,
        min_samples_leaf=10,
        random_state=args.random_state,
        n_jobs=-1,
    )
    kwargs = dict(
        model_y=model_y,
        model_t=model_t,
        n_estimators=args.cf_trees,
        min_samples_leaf=args.min_leaf,
        max_depth=None,
        discrete_treatment=False,
        cv=args.cv,
        random_state=args.random_state,
    )
    try:
        estimator = CausalForestDML(**kwargs, n_jobs=-1)
    except TypeError:
        estimator = CausalForestDML(**kwargs)
    estimator.fit(
        Y=data["Y"][train_idx],
        T=data["T_scaled"][train_idx],
        X=data["X"][train_idx],
        W=data["W"][train_idx],
    )
    return estimator


def get_effect_interval(estimator: CausalForestDML, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        lb, ub = estimator.effect_interval(X)
        return np.asarray(lb).reshape(-1), np.asarray(ub).reshape(-1)
    except Exception as exc:
        print(f"[WARN] effect_interval failed: {exc}")
        return np.full(X.shape[0], np.nan), np.full(X.shape[0], np.nan)


def get_ate_and_interval(estimator: CausalForestDML, X: np.ndarray, cate: np.ndarray) -> tuple[float, float, float]:
    try:
        ate = float(estimator.ate(X))
    except Exception:
        ate = float(np.nanmean(cate))
    try:
        lb, ub = estimator.ate_interval(X)
        return ate, float(lb), float(ub)
    except Exception as exc:
        print(f"[WARN] ate_interval failed: {exc}")
        return ate, np.nan, np.nan


def get_feature_importance(estimator: CausalForestDML, feature_names: list[str]) -> pd.DataFrame:
    try:
        importance = np.asarray(estimator.feature_importances_).reshape(-1)
    except Exception:
        importance = np.full(len(feature_names), np.nan)
    if len(importance) != len(feature_names):
        importance = np.resize(importance, len(feature_names))
    out = pd.DataFrame({"feature": feature_names, "importance": importance})
    out["abs_importance"] = out["importance"].abs()
    return out.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def make_group_summary(values: np.ndarray, cate: np.ndarray, group_type: str) -> pd.DataFrame:
    values_s = pd.Series(values)
    cate_s = pd.Series(cate, dtype=float)
    if group_type == "age":
        groups = pd.cut(pd.to_numeric(values_s, errors="coerce"), [-np.inf, 60, 70, np.inf], labels=["<60", "60-69", ">=70"])
    elif group_type == "bmi":
        groups = pd.cut(pd.to_numeric(values_s, errors="coerce"), [-np.inf, 25, 30, np.inf], labels=["<25", "25-29.9", ">=30"])
    elif group_type == "sex":
        groups = values_s.astype(str)
    else:
        raise ValueError(f"Unsupported group_type: {group_type}")

    temp = pd.DataFrame({"group": groups.astype(str), "CATE_perSD": cate_s})
    temp = temp.replace({"group": {"nan": np.nan, "NaN": np.nan}}).dropna()
    if temp.empty:
        return pd.DataFrame(columns=["group", "n", "mean", "std", "median", "q25", "q75"])
    summary = temp.groupby("group", dropna=False)["CATE_perSD"].agg(["count", "mean", "std", "median"]).reset_index()
    summary = summary.rename(columns={"count": "n"})
    q = temp.groupby("group", dropna=False)["CATE_perSD"].quantile([0.25, 0.75]).unstack().reset_index()
    q = q.rename(columns={0.25: "q25", 0.75: "q75"})
    return summary.merge(q, on="group", how="left")


def make_decile_summary(cate: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"CATE_perSD": pd.to_numeric(pd.Series(cate), errors="coerce")}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["decile", "n", "mean", "std", "median", "q25", "q75"])
    df = df.sort_values("CATE_perSD", ascending=True).reset_index(drop=True)
    df["decile"] = pd.qcut(np.arange(len(df)), q=min(10, len(df)), labels=False, duplicates="drop") + 1
    summary = df.groupby("decile", dropna=False)["CATE_perSD"].agg(["count", "mean", "std", "median"]).reset_index()
    summary = summary.rename(columns={"count": "n"})
    q = df.groupby("decile", dropna=False)["CATE_perSD"].quantile([0.25, 0.75]).unstack().reset_index()
    q = q.rename(columns={0.25: "q25", 0.75: "q75"})
    return summary.merge(q, on="decile", how="left")


def make_age_sex_curve(age: np.ndarray, sex: np.ndarray, cate: np.ndarray, n_bins: int = 8) -> pd.DataFrame:
    temp = pd.DataFrame({
        "Age": pd.to_numeric(pd.Series(age), errors="coerce"),
        "Sex": pd.Series(sex).astype(str),
        "CATE_perSD": pd.to_numeric(pd.Series(cate), errors="coerce"),
    }).dropna()
    rows = []
    for sex_value, group in temp.groupby("Sex", dropna=False):
        if len(group) < 5:
            continue
        group = group.copy()
        try:
            group["age_bin"] = pd.qcut(group["Age"], q=min(n_bins, len(group)), duplicates="drop")
        except Exception:
            continue
        summary = group.groupby("age_bin", observed=False)["CATE_perSD"].agg(["count", "mean", "std"]).reset_index()
        summary = summary.rename(columns={"count": "n"})
        summary["Sex"] = str(sex_value)
        summary["age_mid"] = summary["age_bin"].apply(lambda interval: float(interval.mid))
        rows.append(summary[["Sex", "age_bin", "age_mid", "n", "mean", "std"]])
    if not rows:
        return pd.DataFrame(columns=["Sex", "age_bin", "age_mid", "n", "mean", "std"])
    out = pd.concat(rows, ignore_index=True)
    out["age_bin"] = out["age_bin"].astype(str)
    return out


def compute_top_benefit_stats(cate: np.ndarray, proportions=(0.05, 0.10, 0.20, 0.30, 0.50)) -> dict:
    cate = np.asarray(cate, dtype=float)
    cate = cate[np.isfinite(cate)]
    out = {}
    if len(cate) == 0:
        return out
    sorted_values = np.sort(cate)
    for prop in proportions:
        k = max(1, int(np.floor(len(sorted_values) * prop)))
        out[f"top_{int(prop * 100)}pct_mean_CATE"] = float(np.mean(sorted_values[:k]))
        out[f"top_{int(prop * 100)}pct_n"] = int(k)
    return out


def plot_single_ate(summary: dict, out_dir: Path, save_pdf: bool) -> None:
    ate, lb, ub = summary["ate_perSD"], summary["ate_CI95_low_perSD"], summary["ate_CI95_high_perSD"]
    plt.figure(figsize=(6, 2.8))
    if np.isfinite(lb) and np.isfinite(ub):
        plt.errorbar(ate, [0], xerr=np.array([[max(0, ate - lb)], [max(0, ub - ate)]]), fmt="o", capsize=4)
    else:
        plt.plot([ate], [0], "o")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks([0], [short_treatment_label(summary["Treatment"])])
    plt.xlabel("ATE per SD increase in treatment")
    plt.title("CausalForestDML ATE")
    plt.tight_layout()
    save_current_figure(out_dir / "Figure_ATE_forest_singleT.png", save_pdf=save_pdf)


def plot_cate_distribution(cate: np.ndarray, treatment_col: str, out_dir: Path, save_pdf: bool) -> None:
    cate = np.asarray(cate, dtype=float)
    cate = cate[np.isfinite(cate)]
    if len(cate) == 0:
        return
    plt.figure(figsize=(7, 5))
    plt.hist(cate, bins=40, alpha=0.75)
    plt.axvline(np.mean(cate), linestyle="--", linewidth=1, label="Mean CATE")
    plt.axvline(np.median(cate), linestyle=":", linewidth=1, label="Median CATE")
    plt.axvline(0, linestyle="-.", linewidth=1, label="No effect")
    plt.xlabel("CATE per SD increase in treatment")
    plt.ylabel("Number of individuals")
    plt.title(f"CATE Distribution: {short_treatment_label(treatment_col)}")
    plt.legend()
    plt.tight_layout()
    save_current_figure(out_dir / "Figure_CATE_distribution_perSD.png", save_pdf=save_pdf)


def plot_feature_importance(importance_df: pd.DataFrame, treatment_col: str, out_dir: Path, topk: int, save_pdf: bool) -> None:
    if importance_df.empty:
        return
    plot_df = importance_df.head(topk).iloc[::-1].copy()
    plt.figure(figsize=(8, max(4.5, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Heterogeneity feature")
    plt.title(f"Top Heterogeneity Features: {short_treatment_label(treatment_col)}")
    plt.tight_layout()
    save_current_figure(out_dir / "Figure_feature_importance.png", save_pdf=save_pdf)


def plot_group_summary(summary_df: pd.DataFrame, group_type: str, treatment_col: str, out_dir: Path, save_pdf: bool) -> None:
    if summary_df.empty:
        return
    x = np.arange(len(summary_df))
    means = summary_df["mean"].to_numpy(dtype=float)
    stds = summary_df["std"].fillna(0).to_numpy(dtype=float)
    plt.figure(figsize=(7, 5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, summary_df["group"].astype(str), rotation=30, ha="right")
    plt.ylabel("Mean CATE per SD")
    plt.xlabel(group_type.capitalize())
    plt.title(f"CATE by {group_type.capitalize()}: {short_treatment_label(treatment_col)}")
    plt.tight_layout()
    filename = {"age": "Figure_CATE_subgroup_age.png", "sex": "Figure_CATE_subgroup_sex.png", "bmi": "Figure_CATE_subgroup_BMI.png"}[group_type]
    save_current_figure(out_dir / filename, save_pdf=save_pdf)


def plot_age_sex_curve(curve_df: pd.DataFrame, treatment_col: str, out_dir: Path, save_pdf: bool) -> None:
    if curve_df.empty:
        return
    plt.figure(figsize=(7, 5))
    for sex_value, group in curve_df.groupby("Sex", dropna=False):
        group = group.sort_values("age_mid")
        plt.plot(group["age_mid"], group["mean"], marker="o", label=f"Sex={sex_value}")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Age")
    plt.ylabel("Mean CATE per SD")
    plt.title(f"Age-Sex CATE Curve: {short_treatment_label(treatment_col)}")
    plt.legend()
    plt.tight_layout()
    save_current_figure(out_dir / "Figure_CATE_curve_AgeSex.png", save_pdf=save_pdf)


def plot_decile_summary(decile_df: pd.DataFrame, treatment_col: str, out_dir: Path, save_pdf: bool) -> None:
    if decile_df.empty:
        return
    x = np.arange(len(decile_df))
    plt.figure(figsize=(7, 5))
    plt.bar(x, decile_df["mean"].to_numpy(dtype=float))
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, decile_df["decile"].astype(str))
    plt.xlabel("CATE decile; 1 = most negative estimated effect")
    plt.ylabel("Mean CATE per SD")
    plt.title(f"CATE Decile Summary: {short_treatment_label(treatment_col)}")
    plt.tight_layout()
    save_current_figure(out_dir / "Figure_CATE_decile_summary.png", save_pdf=save_pdf)


def plot_all_treatments_forest(summary_df: pd.DataFrame, out_dir: Path, save_pdf: bool) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.copy()
    plot_df["abs_ate"] = plot_df["ate_perSD"].abs()
    plot_df = plot_df.sort_values("abs_ate", ascending=False).iloc[::-1].reset_index(drop=True)
    plot_df["label"] = plot_df["Treatment"].apply(short_treatment_label)
    y = np.arange(len(plot_df))
    x = plot_df["ate_perSD"].to_numpy(dtype=float)
    lb = plot_df["ate_CI95_low_perSD"].to_numpy(dtype=float)
    ub = plot_df["ate_CI95_high_perSD"].to_numpy(dtype=float)
    plt.figure(figsize=(10.5, max(6.0, 0.45 * len(plot_df))))
    for i in range(len(plot_df)):
        if np.isfinite(lb[i]) and np.isfinite(ub[i]):
            plt.errorbar(x[i], y[i], xerr=np.array([[max(0, x[i] - lb[i])], [max(0, ub[i] - x[i])]]), fmt="o", capsize=3)
        else:
            plt.plot(x[i], y[i], "o")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y, plot_df["label"])
    plt.xlabel("ATE per SD increase in treatment (95% CI)")
    plt.ylabel("DXA-derived bone feature")
    plt.title("CausalForestDML Average Treatment Effects")
    plt.tight_layout()
    save_current_figure(out_dir / "Figure_CausalForestDML_all_treatments_forest.png", save_pdf=save_pdf)


def run_single_treatment(df: pd.DataFrame, treatment_col: str, selected_treatments: list[str], args: argparse.Namespace, x_cols: list[str], w_cols: list[str]) -> dict:
    out_dir = args.out_dir / safe_name(treatment_col)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_single_treatment_data(df, treatment_col, x_cols, w_cols, selected_treatments, args.complete_case)

    if data["n_total"] < args.min_n:
        raise ValueError(f"n after cleaning too small: {data['n_total']}")
    if data["n_unique_t"] < args.min_unique_treatment:
        raise ValueError(f"treatment unique values too few: {data['n_unique_t']}")

    idx = np.arange(data["n_total"])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=maybe_stratify_target(data["Y"]),
    )

    estimator = fit_causal_forest(data, args, train_idx)
    X_test = data["X"][test_idx]
    Y_test = data["Y"][test_idx]
    cate = np.asarray(estimator.effect(X_test)).reshape(-1)
    cate_lb, cate_ub = get_effect_interval(estimator, X_test)
    ate, ate_lb, ate_ub = get_ate_and_interval(estimator, X_test, cate)

    importance_df = get_feature_importance(estimator, data["X_df"].columns.tolist())
    age_test = data["age"][test_idx]
    sex_test = data["sex_raw"][test_idx]
    bmi_test = data["bmi"][test_idx]
    treatment_test_raw = data["T_raw"][test_idx]

    age_summary = make_group_summary(age_test, cate, "age")
    sex_summary = make_group_summary(sex_test, cate, "sex")
    bmi_summary = make_group_summary(bmi_test, cate, "bmi")
    decile_summary = make_decile_summary(cate)
    age_sex_curve = make_age_sex_curve(age_test, sex_test, cate)

    summary = {
        "Treatment": treatment_col,
        "family": classify_family(treatment_col),
        "region": classify_region(treatment_col),
        "n_total": data["n_total"],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_y1": data["n_y1"],
        "n_y0": data["n_y0"],
        "n_unique_treatment": data["n_unique_t"],
        "treatment_sd": data["t_sd"],
        "ate_perSD": ate,
        "ate_CI95_low_perSD": ate_lb,
        "ate_CI95_high_perSD": ate_ub,
        "cate_test_mean": float(np.nanmean(cate)),
        "cate_test_median": float(np.nanmedian(cate)),
        "cate_test_std": float(np.nanstd(cate, ddof=1)) if len(cate) > 1 else np.nan,
        "cate_test_min": float(np.nanmin(cate)),
        "cate_test_max": float(np.nanmax(cate)),
        "top_feature_1": importance_df["feature"].iloc[0] if len(importance_df) else "",
        "top_feature_1_importance": importance_df["importance"].iloc[0] if len(importance_df) else np.nan,
        **compute_top_benefit_stats(cate),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "summary_single_treatment.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    age_summary.to_csv(out_dir / "CATE_subgroup_age.csv", index=False, encoding="utf-8-sig")
    sex_summary.to_csv(out_dir / "CATE_subgroup_sex.csv", index=False, encoding="utf-8-sig")
    bmi_summary.to_csv(out_dir / "CATE_subgroup_BMI.csv", index=False, encoding="utf-8-sig")
    decile_summary.to_csv(out_dir / "CATE_decile_summary.csv", index=False, encoding="utf-8-sig")
    age_sex_curve.to_csv(out_dir / "CATE_curve_age_sex.csv", index=False, encoding="utf-8-sig")

    if not args.no_save_cate_predictions:
        pd.DataFrame({
            "CATE_perSD": cate,
            "CATE_CI95_low_perSD": cate_lb,
            "CATE_CI95_high_perSD": cate_ub,
            "outcome": Y_test,
            "treatment_raw": treatment_test_raw,
            "Age": age_test,
            "Sex": sex_test,
            BMI_COL: bmi_test,
        }).to_csv(out_dir / "CATE_predictions_testset.csv", index=False, encoding="utf-8-sig")

    if not args.no_plots:
        save_pdf = not args.no_pdf
        plot_single_ate(summary, out_dir, save_pdf)
        plot_cate_distribution(cate, treatment_col, out_dir, save_pdf)
        plot_feature_importance(importance_df, treatment_col, out_dir, args.topk_importance, save_pdf)
        plot_group_summary(age_summary, "age", treatment_col, out_dir, save_pdf)
        plot_group_summary(sex_summary, "sex", treatment_col, out_dir, save_pdf)
        plot_group_summary(bmi_summary, "bmi", treatment_col, out_dir, save_pdf)
        plot_age_sex_curve(age_sex_curve, treatment_col, out_dir, save_pdf)
        plot_decile_summary(decile_summary, treatment_col, out_dir, save_pdf)

    return summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df.columns = df.columns.astype(str).str.strip()
    df = deduplicate_columns_keep_first(df)
    df = compute_or_update_bmi(df, skip_recompute=args.skip_bmi_recompute)

    selected_treatments = select_treatments(args.treatments)
    required_cols = build_unique_column_list([OUTCOME_COL] + selected_treatments + CONTROL_COLS + HETEROGENEITY_COLS)
    validate_columns(df, required_cols, context="CausalForestDML analysis")

    print(f"[INFO] Data path: {args.data_path}")
    print(f"[INFO] Data shape: {df.shape}")
    print(f"[INFO] Output directory: {args.out_dir}")
    print(f"[INFO] Number of treatments: {len(selected_treatments)}")
    print(f"[INFO] Complete-case rule: {args.complete_case}")

    summary_rows = []
    failed_rows = []

    for i, treatment_col in enumerate(selected_treatments, start=1):
        print("\n" + "=" * 90)
        print(f"[INFO] {i}/{len(selected_treatments)} Running treatment: {treatment_col}")
        try:
            summary = run_single_treatment(df, treatment_col, selected_treatments, args, HETEROGENEITY_COLS, CONTROL_COLS)
            summary_rows.append(summary)
        except Exception as exc:
            print(f"[ERROR] Failed for {treatment_col}: {exc}")
            failed_rows.append({"Treatment": treatment_col, "error": str(exc), "traceback": traceback.format_exc()})

    summary_df = pd.DataFrame(summary_rows)
    failed_df = pd.DataFrame(failed_rows)

    if not summary_df.empty:
        summary_df["abs_ate_perSD"] = summary_df["ate_perSD"].abs()
        summary_df = summary_df.sort_values("abs_ate_perSD", ascending=False).reset_index(drop=True)
        summary_path = args.out_dir / "CausalForestDML_all_treatments_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] Saved all-treatment summary: {summary_path}")
        if not args.no_plots:
            plot_all_treatments_forest(summary_df, args.out_dir, save_pdf=not args.no_pdf)
    else:
        print("[WARN] No successful treatment results were generated.")

    if not failed_df.empty:
        failed_path = args.out_dir / "CausalForestDML_failed_treatments.csv"
        failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] Saved failed-treatment log: {failed_path}")

    print("[OK] CausalForestDML workflow finished.")


if __name__ == "__main__":
    main()
