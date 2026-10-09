# Dataset Audit

## Scope
This audit reviews the current repository files, the train/test CSVs, and the saved model artifacts before trusting the project for deployment.

## 1. Dataset structure
- Training rows: 45000
- Test rows: 5000
- Columns: ['Student_ID', 'Age', 'Gender', 'Degree', 'Branch', 'CGPA', 'Internships', 'Projects', 'Coding_Skills', 'Communication_Skills', 'Aptitude_Test_Score', 'Soft_Skills_Rating', 'Certifications', 'Backlogs', 'Placement_Status']
- Missing values: {'Student_ID': 0, 'Age': 0, 'Gender': 0, 'Degree': 0, 'Branch': 0, 'CGPA': 0, 'Internships': 0, 'Projects': 0, 'Coding_Skills': 0, 'Communication_Skills': 0, 'Aptitude_Test_Score': 0, 'Soft_Skills_Rating': 0, 'Certifications': 0, 'Backlogs': 0, 'Placement_Status': 0}
- Duplicate rows: 0
- Unique Student_ID values: train=45000, test=5000
- Train target distribution: {'Not Placed': 28688, 'Placed': 16312}
- Test target distribution: {'Not Placed': 3188, 'Placed': 1812}

## 2. Split integrity and overlap
- Overlap in Student_ID between train and test: 0
- Exact record overlap excluding Student_ID: 0
- Shared feature-vector count: 0
- Shared ID preview: []
- Interpretation: the repository preserves a distinct held-out test set; there is no direct record overlap between the files.

## 3. Data quality checks
- Training validation: {'valid': True, 'issues': [], 'shape': (45000, 15), 'row_count': 45000, 'column_count': 15}
- Test validation: {'valid': True, 'issues': [], 'shape': (5000, 15), 'row_count': 5000, 'column_count': 15}
- No missing values were found in the current CSVs, and there are no duplicate identifiers. That said, the dataset can still be synthetic or rule-generated even when it is perfectly clean.

## 4. Likely target-generation behavior
The project notebook is explicit that the target variable is likely generated from deterministic rules and that the Random Forest model reaches near-perfect validation metrics. This is not evidence of a real-world placement process; it instead suggests label construction from a formula based on academic and skill features.

The strongest separation in the training data is visible in the following features:
- CGPA: placed mean 7.65 vs not-placed mean 6.63
- Internships: placed mean 1.12 vs not-placed mean 0.58
- Projects: placed mean 4.35 vs not-placed mean 3.39
- Coding_Skills: placed mean 6.85 vs not-placed mean 5.03
- Communication_Skills: placed mean 6.16 vs not-placed mean 5.13
- Aptitude_Test_Score: placed mean 76.58 vs not-placed mean 65.29
- Backlogs: placed mean 0.26 vs not-placed mean 1.25

Observations:
- CGPA is substantially higher among placed students.
- Backlogs are lower among placed students.
- Coding, communication, aptitude, project, and internship counts all trend upward for placed students.
- The model can separate the classes almost perfectly because the label appears to be strongly tied to these attributes.

## 5. Leakage and feature-risk assessment
- Student_ID is a unique identifier and not a legitimate predictive feature; it should be removed before modeling.
- The dataset does not show obvious leaked target columns, but the label-generation pattern is highly structured and may encode a scoring formula indirectly in the observed feature values.
- Because the target was likely generated from deterministic rules, the resulting accuracy numbers are not trustworthy as real-world deployment metrics.

## 6. Saved model artifact review
The current artifact is a scikit-learn pipeline with a `DecisionTreeClassifier` classifier and 13 input features: ['Age', 'Gender', 'Degree', 'Branch', 'CGPA', 'Internships', 'Projects', 'Coding_Skills', 'Communication_Skills', 'Aptitude_Test_Score', 'Soft_Skills_Rating', 'Certifications', 'Backlogs']. It uses the current scikit-learn 1.8.0 runtime. Rebuild the artifact after dependency changes because joblib pickle compatibility is version-sensitive. Its performance remains limited by the unverified/synthetic target provenance.

## 7. Project readiness assessment
The repository now contains a modular training, preprocessing, prediction, evaluation, validation, explainability, Streamlit, audit, and test implementation. The legacy notebook is isolated to `train.csv` and saves separate demo artifacts. The canonical workflow preserves the original test split and emits current JSON/CSV evaluation outputs.

## 8. Recommended actions
1. Obtain and document a verified real-world dataset before making placement prediction claims.
2. Keep the original train/test CSVs untouched and reserve the test set for the single final evaluation.
3. Consider cross-validation and hyperparameter search on training-only folds only after data provenance is resolved.
4. Treat current scores as pipeline-development results, not as evidence of real-world placement prediction.
5. Use the app only as a demonstration rather than as a real hiring or selection decision tool.
