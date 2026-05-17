# Model Fairness and Explainability with SHAP

## Objectives
- Train a TensorFlow binary classification model on the Adult Income dataset.
- Use SHAP to explain model predictions and global feature importance.
- Evaluate fairness across the sensitive feature `sex`.
- Generate recruiter-ready audit artifacts for explainable and responsible AI.

## Tools Used
- Python 3
- TensorFlow CPU
- SHAP
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- OpenML Adult Income Dataset
- Ubuntu Linux

## Key Skills Demonstrated
- Explainable AI workflow using SHAP
- TensorFlow model training and evaluation
- Fairness auditing using group-level metrics
- Equal opportunity analysis using true positive rate
- Disparate impact analysis using selection rate
- ML preprocessing with one-hot encoding and scaling
- Production-style artifact generation for model audit reports

## Project Outputs
- `fairness_shap_lab.py` — main lab implementation
- `outputs/fairness_metrics_by_sex.csv` — fairness metrics grouped by sex
- `outputs/shap_feature_importance.csv` — ranked SHAP feature importance
- `outputs/shap_summary_plot.png` — visual SHAP explanation plot
- `outputs/model_audit_summary.txt` — plain-text model audit summary

## Troubleshooting Log
- Replaced direct global `pip install` usage with a Python virtual environment to avoid Ubuntu externally-managed Python issues.
- Used `tensorflow-cpu` instead of full TensorFlow GPU package for better compatibility in cloud lab environments.
- Fixed target encoding issue by converting income labels into binary numeric values.
- Fixed fairness grouping bug where income labels were incorrectly used as gender groups.
- Used OpenML dataset loading instead of direct UCI URL dependency to improve reliability.
- Added explicit artifact generation so the lab produces auditable portfolio outputs.
