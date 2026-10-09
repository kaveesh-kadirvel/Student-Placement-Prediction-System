# Final Technical Verification

## Outcome and scope

The canonical script workflow completed successfully. Results below are development-pipeline results on this repository's dataset only. Dataset provenance is unverified; the notebook/audit evidence is consistent with synthetic or rule-generated placement labels. The scores must not be interpreted as real-world placement accuracy or evidence of production readiness.

## Repository and implementation map

| Concern | Canonical file(s) |
|---|---|
| Training and model selection | `scripts/train_models.py`, `src/train.py` |
| Preprocessing | `src/preprocessing.py` |
| Prediction and model loading | `src/predict.py` |
| Metrics | `src/evaluate.py` |
| Feature schema | `src/schema.py` |
| Input/data validation | `src/validation.py` |
| SHAP explanations | `src/explainability.py` |
| Streamlit application | `app.py` |
| Tests | `tests/test_validation.py`, `tests/test_preprocessing.py`, `tests/test_prediction.py`, `tests/test_explainability.py` |
| Audit command | `scripts/audit_dataset.py` |

`project.ipynb` contained a duplicate, older two-model workflow that concatenated `train.csv` and `test.csv`, made another random split, and saved over the canonical model and benchmark artifacts. It was changed to use only `train.csv` and to save notebook-only demo artifacts. The README now identifies it as legacy; `scripts/train_models.py` is the canonical workflow. The README training and app paths match existing files. The direct `streamlit` executable was not on this Windows PATH, so the documented launch command now uses `python -m streamlit`.

## Data integrity and feature schema

- `train.csv`: 45,000 rows; `test.csv`: 5,000 rows; 15 columns in each.
- Neither CSV has missing values or duplicate rows.
- There are zero overlapping `Student_ID` values, zero exact row overlaps excluding the identifier, and zero shared feature vectors.
- `Student_ID` is dropped before fitting. The saved pipeline has 13 input features and excludes the identifier.
- CSV SHA-256 values before and after training were unchanged:
  - `train.csv`: `AA286589D900A8171B8270EC529214D030984910C3368681EB6A0BFB4A58D5D5`
  - `test.csv`: `DE010EF8A41CBCDDACE33F528B695477C59CD5E463068DD4321F3D500E67AC6E`

## Training and evaluation procedure

Executed the README command `python scripts/train_models.py` successfully. All five classifiers fitted successfully. A stratified 80/20 split of `train.csv` is used for validation; preprocessing is fitted within each candidate pipeline on that training partition only. The winning validation candidate is selected by F1, with ROC-AUC and accuracy as tie-breakers. The selected estimator is then refitted on all 45,000 rows of `train.csv`. The original 5,000-row `test.csv` is evaluated only after selection/refitting, and it does not feed preprocessing, model selection, or tuning. No cross-validation or hyperparameter search is implemented; classifier settings are fixed in code.

Important holdout caveat: the canonical training script was successfully rerun several times during this audit while validating fixes, so it recomputed final test metrics on each run. No candidate was chosen or tuned using those test metrics, but the current `test.csv` should not be described as a strictly one-shot, never-revisited evaluation set. For a future independent performance claim, reserve a new untouched test set.

Positive class is `Placed` (encoded as 1). Confusion matrices use rows as actual classes and columns as predicted classes in the order `[Not Placed, Placed]`, i.e. `[[TN, FP], [FN, TP]]`. ROC-AUC is computed from the positive-class probability.

### Validation comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix `[[TN,FP],[FN,TP]]` |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.8603 | 0.7687 | 0.8792 | 0.8202 | 0.9362 | `[[4875,863],[394,2868]]` |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[5738,0],[0,3262]]` |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[5738,0],[0,3262]]` |
| SVM | 0.9843 | 0.9654 | 0.9923 | 0.9787 | 0.9987 | `[[5622,116],[25,3237]]` |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[5738,0],[0,3262]]` |

### Selected model: final held-out test set

The selected model was Decision Tree. On the original held-out `test.csv` (5,000 rows):

| Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix `[[TN,FP],[FN,TP]]` |
|---:|---:|---:|---:|---:|---|
| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[3188,0],[0,1812]]` |

These are actual execution results, but their perfect separation is consistent with rule-generated labels and is not a trustworthy estimate of real-world performance.

Machine-readable outputs are in `reports/model_metrics.json` and `reports/model_metrics.csv`. The JSON includes final test metrics and the validation confusion matrices; the CSV contains the five-model validation comparison.

## SHAP verification

- SHAP 0.53.0 ran successfully against the selected saved pipeline.
- A global explanation was computed for 100 reproducibly sampled training rows; mean absolute SHAP scores and transformed feature names aligned.
- An individual explanation was computed for one profile using a separate, reproducible background sample from `train.csv`. The resulting per-feature attributions were non-zero and their names were members of `preprocessor.get_feature_names_out()`.
- The original implementation failed on SHAP's class-dimension output (`TypeError: float() argument must be a string or a real number, not 'list'`). It was corrected to select the positive-class attribution. A one-row explainer background also produced zero local attributions; it was replaced with training-only background data.
- The app now shows global mean-absolute SHAP values and individual profile SHAP contributions, not ordinary feature importance. The interface states these are model explanations, not causal effects.

## Streamlit and inference verification

- `python -c "import app; print('app-import-ok')"` exited successfully. Bare-mode import emitted Streamlit `ScriptRunContext` warnings, which are expected outside a Streamlit session.
- `python -m streamlit run app.py --server.headless true --server.port 8765` started successfully.
- `curl.exe -sS -i --max-time 15 http://127.0.0.1:8765/_stcore/health` returned HTTP 200 with body `ok`.
- The browser loaded the application, submitted the default valid profile, displayed a prediction and probability, and showed individual SHAP contributions.
- Direct inference returned a valid two-class probability result. A CGPA of 12 was rejected with `ValueError: CGPA must be between 0 and 10`.
- The current browser verification covered all five navigation views, the three overview action buttons, a submitted profile with its class probabilities and individual SHAP contributions, all five comparison rows, global SHAP, and the selected-model held-out confusion-matrix expander. The assessment result remained available after switching screens.
- At 390×844, the page and body widths were both 390 px; navigation wrapped onto a second row and no horizontal page overflow was observed. At the 1920×1080 presentation size, the comparison and prediction views rendered.
- The dashboard and global SHAP Vega-Lite charts render with axes and data, but the browser console still reports Vega warnings: `Infinite extent for field "Students"` and `Infinite extent for field "Mean |SHAP|"`. These are chart warnings, not app startup failures; they remain a UI limitation to investigate. A Streamlit 1.56 `use_container_width` deprecation warning was previously fixed with `width="stretch"`.
- The dashboard does not render the CSS as visible text. The probability labels clarify that they are model estimates and calibration is unverified. Dataset provenance limitations remain visible in the app.
- A current health check using `Invoke-WebRequest -Uri http://localhost:8765/_stcore/health -UseBasicParsing` returned `ok`.

The displayed probability is a classifier probability on synthetic/rule-generated labels, not a calibrated real-world placement chance.

## Tests and dependency environment

Final command: `python -m pytest -q`

- **10 passed**
- **0 failed**
- **0 skipped**

Coverage includes dataset validation, invalid/missing input, Student_ID exclusion and feature schema, model loading/inference, preprocessing, and global/individual SHAP alignment.

Verified environment: Python 3.13.7, pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, Streamlit 1.56.0, XGBoost 3.4.1, SHAP 0.53.0, joblib 1.5.3, pytest 9.1.1. `requirements.txt` pins these tested runtime versions. Matplotlib and Seaborn are declared for the legacy notebook but were not installed in this environment; the notebook's plotting cells were not run. The notebook itself was JSON-validated after the targeted edits.

## Commands actually executed

```powershell
python scripts/train_models.py
python -m pytest -q
python -c "import app; print('app-import-ok')"
python -m streamlit run app.py --server.headless true --server.port 8765
curl.exe -sS -i --max-time 15 http://127.0.0.1:8765/_stcore/health
Invoke-WebRequest -Uri http://localhost:8765/_stcore/health -UseBasicParsing
```

Also executed focused Python checks for model loading, prediction, invalid-input rejection, SHAP global/local outputs, dependency versions, notebook JSON, split overlap, and train/test file hashes.

The first attempt to launch with `streamlit run ...` failed because the executable was not on PATH; the module invocation above worked. The initial SHAP shape/background problems are described above and were fixed before the final passing test run. The training command was executed repeatedly during verification, so each invocation recalculated test metrics; the script and README now warn against using repeated held-out results for tuning.

## Remaining limitations and next steps

1. Dataset provenance and target-generation methodology remain unverified. Obtain and document a legitimate real-world dataset before making predictive-validity claims.
2. Perfect validation and test scores indicate synthetic/rule-generated labels; do not use this model for actual hiring, placement, or student selection decisions.
3. The workflow uses one stratified validation split and fixed model settings; introduce CV/tuning only after establishing valid data provenance and without touching the held-out test set.
4. Install the declared Matplotlib/Seaborn dependencies and execute the notebook only if its legacy exploratory visualizations need to be maintained. It is not the canonical application training workflow.
