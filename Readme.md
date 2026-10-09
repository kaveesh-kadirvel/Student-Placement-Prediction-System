# 🎓 Student Placement Predictor
### AI-Powered Student Placement Prediction & Explainability Platform

<p align="center">
  An interactive machine learning application that predicts student placement outcomes using academic performance, technical skills, internships, aptitude, and other employability indicators.
</p>

<p align="center">
  <a href="https://sies-placement-prediction-system.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Open_Application-6C63FF?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://github.com/kaveesh-kadirvel/Student-Placement-Prediction-System">
    <img src="https://img.shields.io/badge/GitHub-Source_Code-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Classification-189B5A?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/SHAP-Explainable_AI-8A63D2?style=flat-square" alt="SHAP">
</p>

---

## 📌 Overview

**Student Placement Predictor** is a machine learning-based web application developed to analyze student profiles and predict placement outcomes.

The system considers attributes such as CGPA, internships, projects, coding skills, communication skills, aptitude scores, certifications, and academic backlogs to generate a placement prediction.

Beyond prediction, the application provides model comparison, evaluation visualizations, and SHAP-based explanations to help users understand how machine learning models use student attributes.

Built using Python and Streamlit, the project demonstrates an end-to-end machine learning workflow through an interactive web interface.

### 🎯 Project Objectives

- Develop a supervised machine learning system for placement outcome classification.
- Compare multiple classification algorithms.
- Evaluate models using standard classification metrics.
- Provide an interactive interface for entering student information.
- Explore feature contributions using explainable AI.
- Deploy the application online for accessible demonstration.

## 🌐 Live Demo

**Try the application:**  
https://sies-placement-prediction-system.streamlit.app/

**GitHub repository:**  
https://github.com/kaveesh-kadirvel/Student-Placement-Prediction-System

The application includes:

- Interactive student assessment.
- Placement outcome predictions.
- Model comparison and evaluation.
- Confusion matrix visualizations.
- Global feature importance and individual SHAP explanations.
- Project information and responsible-use notes.

---

## ✨ Features

### 🎯 1. Student Placement Prediction

- Interactive student profile assessment form.
- Academic, technical, internship, aptitude, and soft-skill inputs.
- Binary classification into `Placed` or `Not Placed`.
- Model-estimated probability display.
- Individual prediction explanations.

### 🤖 2. Machine Learning Models

The project includes five classification algorithms:

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification baseline |
| Decision Tree | Rule-based classification |
| Random Forest | Ensemble of decision trees |
| Support Vector Machine (SVM) | Margin-based classification |
| XGBoost | Gradient-boosted decision trees |

### 📊 3. Model Evaluation

The application provides visualizations for comparing classification performance using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC, where applicable
- Confusion matrices

These metrics help assess the performance of the evaluated models. Their interpretation depends on the evaluation methodology and dataset quality.

### 🔍 4. Explainable AI with SHAP

The application incorporates SHAP-based explanations to explore model behavior.

**Global explanations**
- Examine which features contribute most across the evaluated data.
- Compare the relative importance of student attributes.

**Individual explanations**
- Inspect feature contributions for a submitted student profile.
- Understand how the model's output relates to the input features.

SHAP explains model behavior and does not establish that a feature causes a placement outcome.

### 🎨 5. Interactive Dashboard

- Modern dark-mode interface.
- Consistent visual styling.
- Organized navigation.
- Student assessment workflow.
- Model comparison charts.
- Explainability visualizations.
- Responsible-use information.

### ☁️ 6. Cloud Deployment

The application is hosted on Streamlit Community Cloud, allowing users to access it through a browser without installing the project locally.

---

## 🧠 Machine Learning Workflow

```text
             Student Dataset
                    │
                    ▼
          Data Inspection & Audit
                    │
                    ▼
       Preprocessing & Feature Selection
                    │
                    ▼
          Train Classification Models
                    │
                    ▼
           Evaluate Model Performance
                    │
                    ▼
          Load Selected Trained Model
                    │
                    ▼
           Student Profile Assessment
                    │
                    ▼
          Placement Outcome Prediction
                    │
                    ▼
          SHAP-Based Model Explanation
```

The workflow covers the main stages of a supervised machine learning project, from dataset preparation and model evaluation to prediction and interpretation.

---

## 📂 Dataset

The project references the **Student Placement Dataset (50,000 Records)**, associated with the report *Data Analysis Documentation: Placement Prediction For IT Students*.

### Dataset Source

| Property | Details |
|---|---|
| Dataset name | Student Placement Dataset (50,000 Records) |
| Reference report | Data Analysis Documentation: Placement Prediction For IT Students |
| Report authors | Aditya Kuril and Neel Kanani |
| Source reference | [Read the matching dataset analysis on Scribd](https://www.scribd.com/document/1038022652/nk-doc) |
| Data format | CSV |
| Total records | 50,000 |
| Target variable | `Placement_Status` |

**Source note:** The linked report is the closest matching source identified for this project. The exact origin of the CSV files in this repository has not been independently confirmed against the original download.

### Dataset Files

| File | Description | Records |
|---|---|---:|
| `train.csv` | Training dataset | 45,000 |
| `test.csv` | Testing dataset | 5,000 |
| `dataset.xlsx` | Additional spreadsheet included in the repository | — |

### Dataset Features

The dataset contains 15 columns representing student attributes and the target placement outcome.

| Feature | Description |
|---|---|
| `Student_ID` | Student identifier |
| `Age` | Student age |
| `Gender` | Gender category |
| `Degree` | Degree category |
| `Branch` | Academic branch |
| `CGPA` | Academic performance indicator |
| `Internships` | Number of internships |
| `Projects` | Number of projects |
| `Coding_Skills` | Coding skills indicator |
| `Communication_Skills` | Communication skills indicator |
| `Aptitude_Test_Score` | Aptitude assessment score |
| `Soft_Skills_Rating` | Soft skills rating |
| `Certifications` | Number of certifications |
| `Backlogs` | Number of academic backlogs |
| `Placement_Status` | Placement classification label |

### Target Variable

The model predicts `Placement_Status`, with two classes:

- `Placed`
- `Not Placed`

The `Student_ID` column is an identifier and should be excluded from predictive model features.

### Dataset Limitations

The dataset may contain synthetic or rule-generated records. Its exact provenance has not been independently established.

Consequently, high evaluation scores should not automatically be interpreted as evidence of real-world placement prediction performance. Further validation on independently sourced data would be needed before applying the model to real student populations.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| XGBoost | Gradient-boosted classification |
| SHAP | Explainable AI |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib / Pickle | Serialized model artifact handling |
| Pytest | Automated testing |
| Git & GitHub | Version control |
| Streamlit Community Cloud | Cloud deployment |

---

## 📁 Project Structure

```text
Student-Placement-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── src/
│   ├── schema.py
│   ├── predict.py
│   ├── explainability.py
│   └── ...
│
├── reports/
│   ├── dataset_audit.md
│   └── model_metrics.json
│
├── tests/
│   └── ...
│
├── train.csv
├── test.csv
├── dataset.xlsx
├── placement_model.pkl
└── project.ipynb
```

*Note: This tree illustrates the main project components. Refer to the repository for the exact current file structure.*

---

## ⚙️ Run the Project Locally

Follow the instructions below to run the application on your computer.

### Prerequisites

- Python 3.13, matching the locally tested environment.
- Git.
- pip.

### 1. Clone the Repository

```bash
git clone https://github.com/kaveesh-kadirvel/Student-Placement-Prediction-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Student-Placement-Prediction-System
```

### 3. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the local address shown in the terminal, usually:

```text
http://localhost:8501
```

### 6. Run Automated Tests

```bash
python -m pytest -q
```

The tests verify selected aspects of the project. Results may vary depending on the current repository version and environment.

---

## 📈 Model Evaluation Metrics

The project uses standard classification metrics to evaluate model performance.

| Metric | Description |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Precision | Proportion of predicted positive cases that are correct |
| Recall | Proportion of actual positive cases identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to rank positive cases above negative cases |

### Confusion Matrix

A confusion matrix summarizes the number of:

- True positives
- True negatives
- False positives
- False negatives

These values help identify different types of classification errors.

**Evaluation note:** Reported scores should be calculated from genuine model predictions using the appropriate held-out evaluation data. Perfect-looking metrics should be interpreted cautiously, especially when the dataset's provenance is uncertain.

---

## 🔐 Responsible Use

This project is an educational machine learning prototype intended for experimentation and demonstration.

- Dataset provenance has not been independently verified.
- The data may be synthetic or generated using predefined rules.
- Model performance on this dataset may not generalize to real student populations.
- Predictions can reflect biases and patterns present in the training data.
- SHAP values describe model behavior rather than causal relationships.
- Model-estimated probabilities are not necessarily calibrated estimates of real-world placement likelihood.
- Predictions should not be used as the sole basis for hiring, admissions, or other consequential decisions.

The application should be treated as a learning and demonstration tool, not a validated employment decision system.

---

## 🚀 Future Enhancements

Potential future improvements include:

- Verify the original dataset source and licensing.
- Validate models on independently sourced real-world data.
- Implement cross-validation and probability calibration.
- Conduct subgroup performance and fairness analysis.
- Add model versioning and experiment tracking.
- Improve automated testing and data validation.
- Introduce data drift and performance monitoring.
- Develop a reproducible model retraining workflow.

---

## 🎓 Academic Project

| Property | Details |
|---|---|
| Project title | Student Placement Predictor Using Machine Learning |
| Domain | Machine Learning and Predictive Analytics |
| Application type | Interactive web application |
| Classification task | Binary classification |
| Explainability | SHAP |
| Deployment | Streamlit Community Cloud |

The project demonstrates the integration of data preprocessing, supervised machine learning, model evaluation, interactive visualization, and explainable AI.

---

## 👨‍💻 Author

**Kaveesh Kadirvel**

- **GitHub:** [@kaveesh-kadirvel](https://github.com/kaveesh-kadirvel)
- **Repository:** [Student Placement Prediction System](https://github.com/kaveesh-kadirvel/Student-Placement-Prediction-System)
- **Live Application:** [Launch Student Placement Predictor](https://sies-placement-prediction-system.streamlit.app/)

---

<p align="center">
  <strong>Built with Python, Machine Learning, and Explainable AI.</strong>
  <br>
  <sub>Academic project · Predictive analytics · Responsible machine learning</sub>
</p>
