# Student Placement Prediction System

This project is a small ML and Streamlit application for predicting student placement outcomes from academic, technical, and soft-skill inputs.

## Important limitation
The repository contains a synthetic or rule-generated dataset rather than a verified real-world placement dataset. The notebook and audit notes show strong feature separation and near-perfect model metrics, which is a sign of deterministic label generation rather than real-world placement behavior. This project should therefore be treated as a reproducible ML pipeline example and educational demo rather than a production placement model.

## Project goals
- Predict whether a student is placed or not placed using campus-relevant attributes
- Compare five classifiers: Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost
- Evaluate models with accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC
- Provide SHAP-based explanations where supported
- Deliver a simplified Streamlit interface for prediction and analytics

## Repository structure
- `app.py` — Streamlit dashboard
- `train.csv` and `test.csv` — dataset split used for model development and final evaluation
- `placement_model.pkl` — saved trained pipeline artifact
- `placed_avg_stats.pkl` — benchmark statistics for placed students
- `scripts/train_models.py` — trains the benchmark models and saves metrics/artifacts
- `scripts/audit_dataset.py` — generates the dataset audit report
- `src/` — modular ML pipeline code
- `reports/` — evaluation and audit outputs
- `tests/` — validation and integration tests

`project.ipynb` is an exploratory legacy notebook. It now uses only `train.csv` for its internal train/validation split and saves separate `*_notebook_demo.pkl` files. Use `scripts/train_models.py` for the canonical workflow and application artifacts.

## Setup
```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train the project
```bash
python scripts/train_models.py
```

## Run the app
```bash
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

The app entry point is `app.py` at the repository root. Community Cloud needs the files below to run the deployed app:

- `app.py`, `src/`, and `requirements.txt`
- `.streamlit/config.toml`
- `placement_model.pkl` (the saved fitted model pipeline)
- `train.csv` (dashboard summaries, form ranges, and SHAP background data)
- `reports/model_metrics.json` (verified evaluation report displayed in the app)

`test.csv` is retained for reproducibility and the training/evaluation workflow, but the deployed app reads its displayed test metrics from the saved report. `dataset.xlsx`, the legacy notebook, and `placed_avg_stats.pkl` are not needed by the app at runtime. The trained model file is explicitly allowed by `.gitignore` because it must be included in the GitHub repository for deployment.

1. Upload this project folder to a GitHub repository you own. This workspace is not currently connected to a Git repository, so no repository, branch, or remote URL can be selected on your behalf. Include the runtime files listed above; do not upload `.env` files, credentials, local virtual environments, caches, or backup folders.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub. Click **Create app**, then **Yup, I have an app**.
3. Select your GitHub repository, the branch containing the project files (commonly `main`), and `app.py` as the main file path.
4. Open **Advanced settings** and select **Python 3.13**, matching the Python version used for local verification. Leave **Secrets** empty; this project does not require API keys or credentials.
5. Click **Deploy**. Wait for the app to finish building, then open the generated `*.streamlit.app` URL.
6. Verify the Overview screen, submit a student assessment, and open Model comparison and Explainability. The first SHAP view may take a short time to calculate. Treat all metrics as development results because dataset provenance is unverified.

If GitHub does not yet contain this project, create a new empty repository in GitHub, then use GitHub Desktop or the GitHub website's upload flow to add the project files. Do not include unrelated backup folders. This workspace has no Git metadata or configured remote, so no changes have been committed or pushed.

## Notes
- The project keeps the original train and test CSV files intact.
- Each training run evaluates the test set only after model selection; rerunning the script recomputes test metrics, so do not use repeated results for tuning or selection.
- All preprocessing is fit on training data only.
- Model performance is reported only after actual execution and review.
