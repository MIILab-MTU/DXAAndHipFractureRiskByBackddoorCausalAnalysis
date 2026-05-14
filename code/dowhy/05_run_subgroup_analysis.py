"""
Run male/female subgroup backdoor causal analysis and create subgroup forest plots.

This script is the cleaned GitHub-ready version of the previous test12_sex.py workflow.
It runs the same backdoor linear-adjustment analysis separately in male and female
participants, then saves one result table and one forest plot for each subgroup.

Default input:
    data/BackdoorData.csv

Default outputs:
    results/dowhy/subgroup/male/backdoor_results_male_perSD_with_CI_and_refuters.csv
    results/dowhy/subgroup/female/backdoor_results_female_perSD_with_CI_and_refuters.csv
    results/figures/subgroup/Figure_Backdoor_ForestPlot_Male_perSD.png/.pdf
    results/figures/subgroup/Figure_Backdoor_ForestPlot_Female_perSD.png/.pdf

Example:
    python scripts/05_run_subgroup_analysis.py \
        --data-path data/BackdoorData.csv \
        --sex-col Sex

If your sex variable is coded numerically, specify the values directly, for example:
    python scripts/05_run_subgroup_analysis.py \
        --data-path data/BackdoorData.csv \
        --sex-col Sex \
        --male-values 1 \
        --female-values 0

If your dataset has already been split into male and female files:
    python scripts/05_run_subgroup_analysis.py \
        --male-data-path data/Backdoor_Male.csv \
        --female-data-path data/Backdoor_Female.csv
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dowhy import CausalModel


warnings.filterwarnings("ignore")


# =============================================================================
# Variable definitions
# =============================================================================

OUTCOME_COL = "hip_fracture"

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

BASE_CONFOUNDER_COLS = [
    "Age",
    "Sex",
    "Height",
    "Weight",
    # "BMI",
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


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run male/female subgroup backdoor causal analysis."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data") / "BackdoorData.csv",
        help=(
            "Path to the full local patient-level dataset. "
            "Used when --male-data-path and --female-data-path are not provided."
        ),
    )
    parser.add_argument(
        "--male-data-path",
        type=Path,
        default=None,
        help="Optional path to an already filtered male dataset.",
    )
    parser.add_argument(
        "--female-data-path",
        type=Path,
        default=None,
        help="Optional path to an already filtered female dataset.",
    )
    parser.add_argument(
        "--sex-col",
        type=str,
        default="Sex",
        help="Column used to define male/female subgroups when using the full dataset.",
    )
    parser.add_argument(
        "--male-values",
        type=str,
        default="Male,M,male,m,1",
        help="Comma-separated values in --sex-col that should be treated as male.",
    )
    parser.add_argument(
        "--female-values",
        type=str,
        default="Female,F,female,f,0",
        help="Comma-separated values in --sex-col that should be treated as female.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "dowhy" / "subgroup",
        help="Directory for subgroup result tables.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("results") / "figures" / "subgroup",
        help="Directory for subgroup forest plots.",
    )
    parser.add_argument(
        "--complete-case",
        choices=["per-treatment", "all"],
        default="per-treatment",
        help=(
            "Missing-data rule. "
            "'per-treatment' drops missing values separately for each treatment. "
            "'all' uses one fixed complete-case cohort across all treatments and confounders."
        ),
    )
    parser.add_argument(
        "--include-unobserved-refuter",
        action="store_true",
        help="Run DoWhy add_unobserved_common_cause refuter for subgroup analyses.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for PNG output.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Save PNG only and skip PDF output.",
    )
    parser.add_argument(
        "--no-value-labels",
        action="store_true",
        help="Do not display numeric estimate and 95% CI labels on subgroup forest plots.",
    )
    return parser.parse_args()


# =============================================================================
# General helper functions
# =============================================================================

def split_value_list(values: str) -> set[str]:
    return {item.strip() for item in values.split(",") if item.strip()}


def validate_columns(
    df: pd.DataFrame,
    required_cols: Iterable[str],
    context: str,
) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing for {context}:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def classify_family(x_name: str) -> str:
    x_low = str(x_name).lower()
    if "t-score" in x_low:
        return "T-score"
    if "bmd" in x_low:
        return "BMD"
    if "bmc" in x_low:
        return "BMC"
    return "Other"


def classify_region(x_name: str) -> str:
    x_low = str(x_name).lower()
    if "neck" in x_low:
        return "Neck"
    if "total" in x_low:
        return "Total"
    if "troch" in x_low:
        return "Troch"
    if "wards" in x_low:
        return "Wards"
    if "shaft" in x_low:
        return "Shaft"
    if "pelvis" in x_low:
        return "Pelvis"
    return "Other"


def shorten_feature_label(x_name: str) -> str:
    x_low = str(x_name).lower()

    if "neck" in x_low:
        region = "Femoral neck"
    elif "total" in x_low:
        region = "Total femur"
    elif "troch" in x_low:
        region = "Trochanter"
    elif "wards" in x_low:
        region = "Ward's region"
    elif "shaft" in x_low:
        region = "Femoral shaft"
    elif "pelvis" in x_low:
        region = "Pelvis"
    else:
        region = "Other region"

    if "t-score" in x_low:
        metric = "T-score"
    elif "bmd" in x_low:
        metric = "BMD"
    elif "bmc" in x_low:
        metric = "BMC"
    else:
        metric = "Other metric"

    return f"{region} {metric}"


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def parse_refuter_pvalue(refuter_obj) -> float:
    if refuter_obj is None:
        return np.nan

    text = str(getattr(refuter_obj, "refutation_result", ""))
    match = re.search(r"p_value\s*:\s*([0-9.eE+-]+)", text)
    if not match:
        return np.nan

    try:
        return float(match.group(1))
    except Exception:
        return np.nan


def load_csv(path: Path, context: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{context} file not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    print(f"[INFO] Loaded {context}: {path}")
    print(f"[INFO] {context} shape: {df.shape}")
    return df


def filter_subgroup_from_full_data(
    df: pd.DataFrame,
    sex_col: str,
    male_values: set[str],
    female_values: set[str],
) -> dict[str, pd.DataFrame]:
    validate_columns(df, [sex_col], context="subgroup filtering")

    sex_as_str = df[sex_col].astype(str).str.strip()

    male_mask = sex_as_str.isin(male_values)
    female_mask = sex_as_str.isin(female_values)

    male_df = df.loc[male_mask].copy()
    female_df = df.loc[female_mask].copy()

    if male_df.empty:
        print(
            "[WARN] Male subgroup is empty. "
            "Check --sex-col and --male-values."
        )

    if female_df.empty:
        print(
            "[WARN] Female subgroup is empty. "
            "Check --sex-col and --female-values."
        )

    print(f"[INFO] Male subgroup rows: {len(male_df)}")
    print(f"[INFO] Female subgroup rows: {len(female_df)}")

    return {
        "Male": male_df,
        "Female": female_df,
    }


# =============================================================================
# Analysis functions
# =============================================================================

def prepare_analysis_dataframe(
    df: pd.DataFrame,
    x_col: str,
    outcome_col: str,
    confounder_cols: list[str],
    complete_case: str,
) -> tuple[pd.DataFrame, list[str], float]:
    if complete_case == "all":
        need_cols = [outcome_col] + TREATMENT_COLS + confounder_cols
        base = df[need_cols].dropna().copy()
        dfi = base[[outcome_col, x_col] + confounder_cols].copy()
    else:
        need_cols = [outcome_col, x_col] + confounder_cols
        dfi = df[need_cols].dropna().copy()

    if dfi.empty:
        return pd.DataFrame(), [], np.nan

    dfi[outcome_col] = pd.to_numeric(dfi[outcome_col], errors="coerce")
    dfi[x_col] = pd.to_numeric(dfi[x_col], errors="coerce")

    sd_x = float(dfi[x_col].std(ddof=0))

    x_and_c = dfi[[x_col] + confounder_cols].copy()
    categorical_cols = x_and_c.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    work = pd.get_dummies(x_and_c, columns=categorical_cols, drop_first=True)
    work = work.apply(pd.to_numeric, errors="coerce")
    work[outcome_col] = dfi[outcome_col]

    work = work.dropna().copy()
    if work.empty:
        return pd.DataFrame(), [], np.nan

    common_causes = [col for col in work.columns if col not in [x_col, outcome_col]]
    return work, common_causes, sd_x


def run_statsmodels_linear_regression(
    work: pd.DataFrame,
    x_col: str,
    outcome_col: str,
    sd_x: float,
) -> dict:
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    x_df = work.drop(columns=[outcome_col]).copy()
    x_df = sm.add_constant(x_df, has_constant="add")

    valid_mask = x_df.notnull().all(axis=1) & y.notnull()
    x_df = x_df.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    if x_df.empty or x_col not in x_df.columns:
        return {
            "estimate_raw": np.nan,
            "stderr_raw": np.nan,
            "p_value_raw": np.nan,
            "CI95_low_raw": np.nan,
            "CI95_high_raw": np.nan,
            "estimate_perSD": np.nan,
            "stderr_perSD": np.nan,
            "CI95_low_perSD": np.nan,
            "CI95_high_perSD": np.nan,
        }

    fit = sm.OLS(y, x_df).fit()

    beta = safe_float(fit.params.get(x_col, np.nan))
    se = safe_float(fit.bse.get(x_col, np.nan))
    p_value = safe_float(fit.pvalues.get(x_col, np.nan))

    try:
        ci_low, ci_high = fit.conf_int(alpha=0.05).loc[x_col].tolist()
        ci_low = safe_float(ci_low)
        ci_high = safe_float(ci_high)
    except Exception:
        ci_low, ci_high = np.nan, np.nan

    return {
        "estimate_raw": beta,
        "stderr_raw": se,
        "p_value_raw": p_value,
        "CI95_low_raw": ci_low,
        "CI95_high_raw": ci_high,
        "estimate_perSD": beta * sd_x if np.isfinite(beta) and np.isfinite(sd_x) else np.nan,
        "stderr_perSD": se * sd_x if np.isfinite(se) and np.isfinite(sd_x) else np.nan,
        "CI95_low_perSD": ci_low * sd_x if np.isfinite(ci_low) and np.isfinite(sd_x) else np.nan,
        "CI95_high_perSD": ci_high * sd_x if np.isfinite(ci_high) and np.isfinite(sd_x) else np.nan,
    }


def run_dowhy_analysis(
    work: pd.DataFrame,
    x_col: str,
    outcome_col: str,
    common_causes: list[str],
    sd_x: float,
    include_unobserved_refuter: bool,
) -> dict:
    model = CausalModel(
        data=work,
        treatment=x_col,
        outcome=outcome_col,
        common_causes=common_causes,
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )

    dowhy_estimate_raw = safe_float(getattr(estimate, "value", np.nan))
    dowhy_estimate_per_sd = (
        dowhy_estimate_raw * sd_x
        if np.isfinite(dowhy_estimate_raw) and np.isfinite(sd_x)
        else np.nan
    )

    ref_placebo = None
    ref_random = None
    ref_unobserved = None

    try:
        ref_placebo = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
        )
    except Exception as exc:
        print(f"[WARN] Placebo refuter failed for {x_col}: {exc}")

    try:
        ref_random = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause",
        )
    except Exception as exc:
        print(f"[WARN] Random common cause refuter failed for {x_col}: {exc}")

    if include_unobserved_refuter:
        try:
            ref_unobserved = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                confounders_effect_on_treatment="binary_flip",
                confounders_effect_on_outcome="binary_flip",
                effect_strength_on_treatment=0.1,
                effect_strength_on_outcome=0.1,
            )
        except Exception as exc:
            print(f"[WARN] Unobserved common cause refuter failed for {x_col}: {exc}")

    placebo_raw = (
        safe_float(getattr(ref_placebo, "new_effect", np.nan))
        if ref_placebo is not None
        else np.nan
    )
    random_raw = (
        safe_float(getattr(ref_random, "new_effect", np.nan))
        if ref_random is not None
        else np.nan
    )
    unobserved_raw = (
        safe_float(getattr(ref_unobserved, "new_effect", np.nan))
        if ref_unobserved is not None
        else np.nan
    )

    return {
        "identified_estimand": str(identified_estimand),
        "dowhy_estimate_raw": dowhy_estimate_raw,
        "dowhy_estimate_perSD": dowhy_estimate_per_sd,
        "placebo_new_effect_raw": placebo_raw,
        "placebo_new_effect_perSD": (
            placebo_raw * sd_x if np.isfinite(placebo_raw) and np.isfinite(sd_x) else np.nan
        ),
        "placebo_p_value": parse_refuter_pvalue(ref_placebo),
        "placebo_refutation_result": (
            str(getattr(ref_placebo, "refutation_result", ""))
            if ref_placebo is not None
            else ""
        ),
        "random_cc_new_effect_raw": random_raw,
        "random_cc_new_effect_perSD": (
            random_raw * sd_x if np.isfinite(random_raw) and np.isfinite(sd_x) else np.nan
        ),
        "random_cc_p_value": parse_refuter_pvalue(ref_random),
        "random_cc_refutation_result": (
            str(getattr(ref_random, "refutation_result", ""))
            if ref_random is not None
            else ""
        ),
        "unobserved_cc_new_effect_raw": unobserved_raw,
        "unobserved_cc_new_effect_perSD": (
            unobserved_raw * sd_x
            if np.isfinite(unobserved_raw) and np.isfinite(sd_x)
            else np.nan
        ),
        "unobserved_cc_p_value": parse_refuter_pvalue(ref_unobserved),
        "unobserved_cc_refutation_result": (
            str(getattr(ref_unobserved, "refutation_result", ""))
            if ref_unobserved is not None
            else ""
        ),
    }


def run_subgroup_analysis(
    subgroup_df: pd.DataFrame,
    subgroup_name: str,
    out_dir: Path,
    complete_case: str,
    include_unobserved_refuter: bool,
) -> pd.DataFrame:
    subgroup_out_dir = out_dir / subgroup_name.lower()
    subgroup_out_dir.mkdir(parents=True, exist_ok=True)

    # Sex is the subgroup-defining variable and should not be adjusted for
    # within sex-stratified analyses.
    confounder_cols = [col for col in BASE_CONFOUNDER_COLS if col != "Sex"]

    required_cols = [OUTCOME_COL] + TREATMENT_COLS + confounder_cols
    validate_columns(subgroup_df, required_cols, context=f"{subgroup_name} subgroup analysis")

    rows = []

    for idx, x_col in enumerate(TREATMENT_COLS, start=1):
        print("\n" + "=" * 90)
        print(f"[INFO] {subgroup_name}: {idx}/{len(TREATMENT_COLS)} treatment = {x_col}")

        work, common_causes, sd_x = prepare_analysis_dataframe(
            df=subgroup_df,
            x_col=x_col,
            outcome_col=OUTCOME_COL,
            confounder_cols=confounder_cols,
            complete_case=complete_case,
        )

        if work.empty:
            print(f"[WARN] {subgroup_name}: no usable data for {x_col}. Skipping.")
            continue

        print(f"[INFO] {subgroup_name}: n_used = {len(work)}")
        print(f"[INFO] {subgroup_name}: n_common_causes_after_dummy = {len(common_causes)}")
        print(f"[INFO] {subgroup_name}: sd_X = {sd_x:.6f}")

        try:
            dowhy_result = run_dowhy_analysis(
                work=work,
                x_col=x_col,
                outcome_col=OUTCOME_COL,
                common_causes=common_causes,
                sd_x=sd_x,
                include_unobserved_refuter=include_unobserved_refuter,
            )
        except Exception as exc:
            print(f"[ERROR] {subgroup_name}: DoWhy analysis failed for {x_col}: {exc}")
            continue

        sm_result = run_statsmodels_linear_regression(
            work=work,
            x_col=x_col,
            outcome_col=OUTCOME_COL,
            sd_x=sd_x,
        )

        rows.append(
            {
                "subgroup": subgroup_name,
                "X_variable": x_col,
                "family": classify_family(x_col),
                "region": classify_region(x_col),
                "n_used": int(len(work)),
                "n_common_causes_after_dummy": int(len(common_causes)),
                "sd_X": sd_x,
                **sm_result,
                **dowhy_result,
            }
        )

    results = pd.DataFrame(rows)

    if results.empty:
        print(f"[WARN] {subgroup_name}: no results generated.")
        return results

    results["abs_estimate_perSD"] = results["estimate_perSD"].abs()
    results = results.sort_values(
        by=["family", "abs_estimate_perSD"],
        ascending=[True, False],
    ).reset_index(drop=True)

    out_csv = subgroup_out_dir / f"backdoor_results_{subgroup_name.lower()}_perSD_with_CI_and_refuters.csv"
    results.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n[OK] {subgroup_name}: saved subgroup results:")
    print(out_csv)

    return results


# =============================================================================
# Plot functions
# =============================================================================

def create_subgroup_forest_plot(
    results: pd.DataFrame,
    subgroup_name: str,
    fig_dir: Path,
    dpi: int,
    save_pdf: bool,
    show_value_labels: bool,
) -> None:
    if results.empty:
        print(f"[WARN] {subgroup_name}: no results available for forest plot.")
        return

    required_cols = [
        "X_variable",
        "family",
        "estimate_perSD",
        "CI95_low_perSD",
        "CI95_high_perSD",
    ]
    validate_columns(results, required_cols, context=f"{subgroup_name} forest plot")

    plot_df = results[required_cols].copy()
    for col in ["estimate_perSD", "CI95_low_perSD", "CI95_high_perSD"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.dropna(subset=["estimate_perSD", "CI95_low_perSD", "CI95_high_perSD"]).copy()
    if plot_df.empty:
        print(f"[WARN] {subgroup_name}: no valid rows remain for forest plot.")
        return

    plot_df["label"] = plot_df["X_variable"].apply(shorten_feature_label)
    plot_df["abs_estimate_perSD"] = plot_df["estimate_perSD"].abs()
    plot_df = plot_df.sort_values("abs_estimate_perSD", ascending=False).reset_index(drop=True)
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    color_map = {
        "BMC": "#f4a6a6",
        "BMD": "#8ecae6",
        "T-score": "#a8ddb5",
        "Other": "#cccccc",
    }

    x = plot_df["estimate_perSD"].to_numpy(dtype=float)
    lo = plot_df["CI95_low_perSD"].to_numpy(dtype=float)
    hi = plot_df["CI95_high_perSD"].to_numpy(dtype=float)

    xerr_left = x - lo
    xerr_right = hi - x

    y = np.arange(len(plot_df))
    fig_height = max(6.0, 0.45 * len(plot_df))

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, fig_height))

    point_colors = plot_df["family"].map(color_map).fillna(color_map["Other"]).tolist()

    for i in range(len(plot_df)):
        plt.errorbar(
            x[i],
            y[i],
            xerr=np.array([[xerr_left[i]], [xerr_right[i]]]),
            fmt="o",
            color=point_colors[i],
            ecolor=point_colors[i],
            elinewidth=2,
            capsize=3,
            markersize=6,
        )

    plt.axvline(0, linestyle="--", color="black", linewidth=1)

    plt.yticks(y, plot_df["label"].values, fontsize=10)
    plt.xlabel("Causal effect per SD increase (95% CI)", fontsize=12)
    plt.ylabel("DXA-derived bone feature", fontsize=12)
    plt.title(
        f"Backdoor Causal Effects of DXA-Derived Bone Features on Hip Fracture ({subgroup_name})",
        fontsize=14,
    )

    if show_value_labels:
        finite_x = x[np.isfinite(x)]
        x_range = np.nanmax(np.abs(finite_x)) if len(finite_x) > 0 else 1.0
        offset = 0.02 * x_range if x_range > 0 else 0.001

        for i, value in enumerate(x):
            text = f"{value:.4f} [{lo[i]:.4f}, {hi[i]:.4f}]"
            if value >= 0:
                plt.text(value + offset, y[i], text, va="center", fontsize=8)
            else:
                plt.text(value - offset, y[i], text, va="center", ha="right", fontsize=8)

    legend_elements = [
        Patch(facecolor=color_map["BMC"], label="BMC"),
        Patch(facecolor=color_map["BMD"], label="BMD"),
        Patch(facecolor=color_map["T-score"], label="T-score"),
    ]
    plt.legend(handles=legend_elements, title="Feature family", loc="lower right")

    plt.tight_layout()

    png_path = fig_dir / f"Figure_Backdoor_ForestPlot_{subgroup_name}_perSD.png"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] {subgroup_name}: saved PNG: {png_path}")

    if save_pdf:
        pdf_path = fig_dir / f"Figure_Backdoor_ForestPlot_{subgroup_name}_perSD.pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"[OK] {subgroup_name}: saved PDF: {pdf_path}")

    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.male_data_path is not None or args.female_data_path is not None:
        if args.male_data_path is None or args.female_data_path is None:
            raise ValueError(
                "Please provide both --male-data-path and --female-data-path, "
                "or provide neither and use --data-path with --sex-col."
            )

        subgroup_data = {
            "Male": load_csv(args.male_data_path, "male subgroup dataset"),
            "Female": load_csv(args.female_data_path, "female subgroup dataset"),
        }
    else:
        full_df = load_csv(args.data_path, "full dataset")
        subgroup_data = filter_subgroup_from_full_data(
            df=full_df,
            sex_col=args.sex_col,
            male_values=split_value_list(args.male_values),
            female_values=split_value_list(args.female_values),
        )

    combined_results = []

    for subgroup_name in ["Male", "Female"]:
        subgroup_df = subgroup_data[subgroup_name]

        if subgroup_df.empty:
            print(f"[WARN] {subgroup_name}: subgroup dataset is empty. Skipping.")
            continue

        results = run_subgroup_analysis(
            subgroup_df=subgroup_df,
            subgroup_name=subgroup_name,
            out_dir=args.out_dir,
            complete_case=args.complete_case,
            include_unobserved_refuter=args.include_unobserved_refuter,
        )

        create_subgroup_forest_plot(
            results=results,
            subgroup_name=subgroup_name,
            fig_dir=args.fig_dir,
            dpi=args.dpi,
            save_pdf=not args.no_pdf,
            show_value_labels=not args.no_value_labels,
        )

        if not results.empty:
            combined_results.append(results)

    if combined_results:
        combined_df = pd.concat(combined_results, ignore_index=True)
        combined_path = args.out_dir / "backdoor_results_male_female_combined_perSD_with_CI_and_refuters.csv"
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved combined male/female results: {combined_path}")

    print("[OK] Subgroup analysis finished.")


if __name__ == "__main__":
    main()
