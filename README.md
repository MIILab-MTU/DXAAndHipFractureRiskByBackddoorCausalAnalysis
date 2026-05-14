# DXA and Hip Fracture Risk by Backdoor Causal Analysis

This repository contains code for DAG-guided backdoor causal analysis of DXA-derived skeletal features and hip fracture risk.

## Data Availability

Patient-level data are not included in this repository due to data use agreements, privacy restrictions, and institutional requirements. The `data/` folder is kept only as a placeholder for local analyses.

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
    Placeholder folder for future CausalForestDML analysis scripts.

  predictive_modeling/
    Placeholder folder for future predictive modeling scripts.

data/
  Local data folder. Patient-level data are not tracked by Git.

results/
  Local output folder. Generated results are not tracked by Git.
```

## DoWhy Backdoor Analysis

The current DoWhy analysis scripts include:

1. `01_run_backdoor_analysis.py`  
   Runs DAG-guided backdoor causal effect estimation and refutation tests.

2. `02_make_main_forest_plot.py`  
   Generates the main causal effect forest plot.

3. `03_make_refutation_figure.py`  
   Generates the refutation comparison figure.

4. `04_make_summary_figures.py`  
   Generates family-level, region-level, and heatmap summary figures.

5. `05_run_subgroup_analysis.py`  
   Runs sex-specific subgroup analyses.

## Requirements

The main Python dependencies are listed in `requirements.txt`.

To install them, use:

```bash
pip install -r requirements.txt
```

## Notes

This repository is intended for code sharing and reproducibility. Restricted patient-level data and generated result files should not be uploaded to GitHub.

## CausalForestDML Analysis

The CausalForestDML scripts include:

1. code/causalforestdml/01_run_causal_forest_dml.py  
   Runs CausalForestDML analysis for all DXA-derived skeletal phenotypes and generates ATE, CATE, subgroup, and feature-importance outputs.

2. code/causalforestdml/02_make_panel_figures.py  
   Combines per-treatment CausalForestDML figures into 4x4 all-treatment panel figures.
