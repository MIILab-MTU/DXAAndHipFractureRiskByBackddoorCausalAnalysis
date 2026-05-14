# DXA and Hip Fracture Risk by Backdoor Causal Analysis

This repository contains analysis code for studying DXA-derived bone features and hip fracture risk using backdoor causal analysis, CausalForestDML heterogeneity analysis, and predictive modeling.

## Data Availability

The patient-level data used in this study cannot be shared publicly due to data use restrictions and participant confidentiality requirements. This repository provides analysis code only. Researchers with approved access to the corresponding datasets may run the scripts locally after placing the approved data in the local `data/` directory.

Patient-level data, intermediate results, model outputs, and individual-level prediction or CATE files are not included in this repository.

## Repository Structure

```text
code/
    dowhy/
        01_run_backdoor_analysis.py
        02_make_main_forest_plot.py
        03_make_refutation_figure.py
        04_make_summary_figures.py
        05_run_subgroup_analysis.py

    causalforestdml/
        01_run_causal_forest_dml.py
        02_make_panel_figures.py

    predictive_modeling/
        01_run_predictive_model_comparison.py
        02_run_logistic_regression_feature_set_comparison.py
        03_run_pure_causal_topk_comparison.py

data/
    Local data folder. Patient-level data are not tracked by Git.

results/
    Local output folder. Results are not tracked by Git.
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## DoWhy Backdoor Analysis

Run the main backdoor analysis:

```bash
python code/dowhy/01_run_backdoor_analysis.py --data-path data/BackdoorData.csv --out-dir results/dowhy
```

Create the main forest plot:

```bash
python code/dowhy/02_make_main_forest_plot.py
```

Create the refutation comparison figure:

```bash
python code/dowhy/03_make_refutation_figure.py
```

Create family, region, and heatmap summary figures:

```bash
python code/dowhy/04_make_summary_figures.py
```

Run male/female subgroup analysis:

```bash
python code/dowhy/05_run_subgroup_analysis.py --data-path data/BackdoorData.csv --sex-col Sex
```

## CausalForestDML Heterogeneity Analysis

Run the CausalForestDML analysis:

```bash
python code/causalforestdml/01_run_causal_forest_dml.py --data-path data/BackdoorData.csv --out-dir results/causal_forest
```

Create 4x4 panel figures from the per-treatment outputs:

```bash
python code/causalforestdml/02_make_panel_figures.py --input-dir results/causal_forest --out-dir results/causal_forest/panels
```

The CausalForestDML workflow estimates average treatment effects and conditional average treatment effects for the prespecified DXA-derived bone features. It also generates CATE distribution plots, heterogeneity feature-importance plots, age/sex/BMI subgroup summaries, age-sex CATE curves, and CATE decile summaries.

## Predictive Modeling

Run the multi-classifier feature-set comparison:

```bash
python code/predictive_modeling/01_run_predictive_model_comparison.py --data-dir data/predictivemodel --out-dir results/predictive_modeling/model_comparison
```

Run the Logistic Regression feature-set comparison:

```bash
python code/predictive_modeling/02_run_logistic_regression_feature_set_comparison.py --data-dir data/predictivemodel --out-dir results/predictive_modeling/logistic_regression_feature_sets
```

Run the pure causal Top-K feature comparison:

```bash
python code/predictive_modeling/03_run_pure_causal_topk_comparison.py --data-path data/predictivemodel/all_features.csv --out-dir results/predictive_modeling/pure_causal_topk
```

## Notes

The `data/` and `results/` folders are excluded from version control. Do not upload patient-level data, raw datasets, intermediate patient-level files, or model output files to GitHub.