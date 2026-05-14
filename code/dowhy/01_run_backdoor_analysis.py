"""
Run DAG-guided backdoor causal analysis for DXA-derived bone features and hip fracture.

This script combines:
1. DoWhy backdoor effect estimation and refutation tests.
2. statsmodels linear regression output for standard errors, p-values, and 95% confidence intervals.

Important:
- Patient-level data should NOT be uploaded to GitHub.
- Place the approved local dataset in a local data directory, or pass the path with --data-path.
- Outputs are written to the results directory and should usually be excluded from GitHub if they contain patient-level information.

Example:
    python scripts/01_run_backdoor_analysis.py \
        --data-path data/BackdoorData.csv \
        --out-dir results/dowhy

Optional:
    python scripts/01_run_backdoor_analysis.py \
        --data-path data/BackdoorData.csv \
        --out-dir results/dowhy \
        --complete-case all
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

CONFOUNDER_COLS = [
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
# Helper functions
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backdoor causal analysis for DXA features and hip fracture."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data") / "BackdoorData.csv",
        help="Path to the local patient-level dataset. This file should not be uploaded to GitHub.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "dowhy",
        help="Directory for output CSV files.",
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
        help="Run DoWhy add_unobserved_common_cause refuter. This can be slower and is treated as sensitivity analysis.",
    )
    return parser.parse_args()


def validate_columns(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_cols: Iterable[str],
    confounder_cols: Iterable[str],
) -> None:
    """Raise an error if required variables are missing."""
    required = [outcome_col] + list(treatment_cols) + list(confounder_cols)
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "The following required columns are missing from the dataset:\n"
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


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def parse_refuter_pvalue(refuter_obj) -> float:
    """Extract p-value from a DoWhy refuter result, if available."""
    if refuter_obj is None:
        return np.nan

    txt = str(getattr(refuter_obj, "refutation_result", ""))
    match = re.search(r"p_value\s*:\s*([0-9.eE+-]+)", txt)
    if not match:
        return np.nan

    try:
        return float(match.group(1))
    except Exception:
        return np.nan


def prepare_analysis_dataframe(
    df: pd.DataFrame,
    x_col: str,
    outcome_col: str,
    confounder_cols: list[str],
    complete_case: str = "per-treatment",
) -> tuple[pd.DataFrame, list[str], float]:
    """
    Prepare a clean dataframe for one treatment.

    Returns:
        work:
            Numeric dataframe containing treatment, outcome, and dummy-encoded confounders.
        common_causes:
            Names of dummy-encoded confounders for DoWhy.
        sd_x:
            Standard deviation of the treatment in the analysis sample.
    """
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
    """
    Fit OLS outcome ~ treatment + confounders.

    This is used to obtain standard error, p-value, and confidence interval
    for the backdoor linear regression estimate.
    """
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    x_df = work.drop(columns=[outcome_col]).copy()
    x_df = sm.add_constant(x_df, has_constant="add")

    valid_mask = x_df.notnull().all(axis=1) & y.notnull()
    x_df = x_df.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    if x_df.empty or x_col not in x_df.columns:
        return {
            "sm_estimate_raw": np.nan,
            "sm_stderr_raw": np.nan,
            "sm_p_value_raw": np.nan,
            "sm_CI95_low_raw": np.nan,
            "sm_CI95_high_raw": np.nan,
            "sm_estimate_perSD": np.nan,
            "sm_stderr_perSD": np.nan,
            "sm_CI95_low_perSD": np.nan,
            "sm_CI95_high_perSD": np.nan,
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
        "sm_estimate_raw": beta,
        "sm_stderr_raw": se,
        "sm_p_value_raw": p_value,
        "sm_CI95_low_raw": ci_low,
        "sm_CI95_high_raw": ci_high,
        "sm_estimate_perSD": beta * sd_x if np.isfinite(beta) and np.isfinite(sd_x) else np.nan,
        "sm_stderr_perSD": se * sd_x if np.isfinite(se) and np.isfinite(sd_x) else np.nan,
        "sm_CI95_low_perSD": ci_low * sd_x if np.isfinite(ci_low) and np.isfinite(sd_x) else np.nan,
        "sm_CI95_high_perSD": ci_high * sd_x if np.isfinite(ci_high) and np.isfinite(sd_x) else np.nan,
    }


def run_dowhy_analysis(
    work: pd.DataFrame,
    x_col: str,
    outcome_col: str,
    common_causes: list[str],
    sd_x: float,
    include_unobserved_refuter: bool = False,
) -> dict:
    """Run DoWhy identify-estimate-refute workflow."""
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


def save_summary_tables(results: pd.DataFrame, out_dir: Path) -> None:
    """Save family-level and region-level summary tables."""
    family_summary = (
        results.groupby("family", dropna=False)
        .agg(
            n_features=("X_variable", "count"),
            mean_estimate_perSD=("estimate_perSD", "mean"),
            mean_abs_estimate_perSD=("estimate_perSD", lambda s: np.mean(np.abs(s))),
            mean_CI95_low_perSD=("CI95_low_perSD", "mean"),
            mean_CI95_high_perSD=("CI95_high_perSD", "mean"),
        )
        .reset_index()
    )

    region_summary = (
        results.groupby("region", dropna=False)
        .agg(
            n_features=("X_variable", "count"),
            mean_estimate_perSD=("estimate_perSD", "mean"),
            mean_abs_estimate_perSD=("estimate_perSD", lambda s: np.mean(np.abs(s))),
            mean_CI95_low_perSD=("CI95_low_perSD", "mean"),
            mean_CI95_high_perSD=("CI95_high_perSD", "mean"),
        )
        .reset_index()
    )

    family_path = out_dir / "backdoor_family_summary.csv"
    region_path = out_dir / "backdoor_region_summary.csv"

    family_summary.to_csv(family_path, index=False, encoding="utf-8-sig")
    region_summary.to_csv(region_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved family summary: {family_path}")
    print(f"[OK] Saved region summary: {region_path}")


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    args = parse_args()

    data_path = args.data_path
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please place the approved local dataset in the data directory, "
            "or pass the path using --data-path."
        )

    print("[INFO] Loading data...")
    df = pd.read_csv(data_path)
    df.columns = df.columns.astype(str).str.strip()

    print(f"[INFO] Data path: {data_path}")
    print(f"[INFO] Data shape: {df.shape}")

    validate_columns(df, OUTCOME_COL, TREATMENT_COLS, CONFOUNDER_COLS)

    rows = []

    for idx, x_col in enumerate(TREATMENT_COLS, start=1):
        print("\n" + "=" * 90)
        print(f"[INFO] {idx}/{len(TREATMENT_COLS)} Running treatment: {x_col}")

        work, common_causes, sd_x = prepare_analysis_dataframe(
            df=df,
            x_col=x_col,
            outcome_col=OUTCOME_COL,
            confounder_cols=CONFOUNDER_COLS,
            complete_case=args.complete_case,
        )

        if work.empty:
            print(f"[WARN] No usable data for {x_col}. Skipping.")
            continue

        print(f"[INFO] n_used = {len(work)}")
        print(f"[INFO] n_common_causes_after_dummy = {len(common_causes)}")
        print(f"[INFO] sd_X = {sd_x:.6f}")

        try:
            dowhy_result = run_dowhy_analysis(
                work=work,
                x_col=x_col,
                outcome_col=OUTCOME_COL,
                common_causes=common_causes,
                sd_x=sd_x,
                include_unobserved_refuter=args.include_unobserved_refuter,
            )
        except Exception as exc:
            print(f"[ERROR] DoWhy analysis failed for {x_col}: {exc}")
            continue

        sm_result = run_statsmodels_linear_regression(
            work=work,
            x_col=x_col,
            outcome_col=OUTCOME_COL,
            sd_x=sd_x,
        )

        # Use statsmodels estimate and CI as the main reported estimate.
        # DoWhy backdoor.linear_regression should match closely because both are linear adjustment.
        row = {
            "X_variable": x_col,
            "family": classify_family(x_col),
            "region": classify_region(x_col),
            "n_used": int(len(work)),
            "n_common_causes_after_dummy": int(len(common_causes)),
            "sd_X": sd_x,
            "estimate_raw": sm_result["sm_estimate_raw"],
            "stderr_raw": sm_result["sm_stderr_raw"],
            "p_value_raw": sm_result["sm_p_value_raw"],
            "CI95_low_raw": sm_result["sm_CI95_low_raw"],
            "CI95_high_raw": sm_result["sm_CI95_high_raw"],
            "estimate_perSD": sm_result["sm_estimate_perSD"],
            "stderr_perSD": sm_result["sm_stderr_perSD"],
            "CI95_low_perSD": sm_result["sm_CI95_low_perSD"],
            "CI95_high_perSD": sm_result["sm_CI95_high_perSD"],
            **dowhy_result,
        }
        rows.append(row)

    results = pd.DataFrame(rows)

    if results.empty:
        print("[WARN] No results were generated.")
        return

    results["abs_estimate_perSD"] = results["estimate_perSD"].abs()
    results = results.sort_values(
        by=["family", "abs_estimate_perSD"],
        ascending=[True, False],
    ).reset_index(drop=True)

    out_csv = out_dir / "backdoor_results_perSD_with_CI_and_refuters.csv"
    results.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n[OK] Saved main results:")
    print(out_csv)
    print("\n[INFO] Preview:")
    print(results[[
        "X_variable",
        "family",
        "region",
        "n_used",
        "estimate_perSD",
        "CI95_low_perSD",
        "CI95_high_perSD",
        "dowhy_estimate_perSD",
    ]].head())

    save_summary_tables(results, out_dir)


if __name__ == "__main__":
    main()
