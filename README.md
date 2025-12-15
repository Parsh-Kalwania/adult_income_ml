# Adult Census Income Analysis (Power BI + Machine Learning)

This project performs an end-to-end analysis of the UCI Adult Census Income dataset, combining data visualization using Power BI with predictive modeling using machine learning.

The objective is to analyze income distribution and identify key factors that influence whether an individual earns more than $50K per year.

---

## Dataset

- Source: UCI Machine Learning Repository (Adult / Census Income dataset)
- Each record represents an individual from the US census
- Target variable: `income` (<=50K, >50K)

---

## Project Structure

adult_income_ml/
│
├── data/
│ └── adult.csv
│
├── src/
│ ├── preprocess.py
│ ├── train_models.py
│ ├── evaluate_best_model.py
│ └── tune_model.py
│
├── requirements.txt
└── README.md


---

## Data Preprocessing

- Removed missing values
- Cleaned target labels (`<=50K`, `>50K`)
- Dropped non-informative column (`fnlwgt`)
- Applied:
  - Standard Scaling for numeric features
  - One-Hot Encoding for categorical features
- Used stratified train-test split to preserve class distribution

---

## Machine Learning Models Used

The following classification models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Model Performance Summary

Gradient Boosting achieved the best overall performance:

- Accuracy: ~87%
- F1-score: ~0.71 (after tuning)
- ROC-AUC: ~0.92

Hyperparameter tuning further improved model performance.

---

## Feature Importance Insights

Key predictors of high income include:
- Marital status (Married-civ-spouse)
- Education level
- Capital gain
- Age
- Working hours per week
- Occupation type

These insights align with trends observed in the Power BI dashboard.

---

## Power BI Dashboard

An interactive Power BI dashboard was built to provide descriptive analysis, including:
- Income distribution
- Demographic breakdown
- Education and occupation analysis
- Work and investment patterns

The dashboard complements the machine learning models by providing visual insights.

---

## How to Run the Project

1. Install dependencies:
bash
pip install -r requirements.txt

2.Train and compare models:

python src/train_models.py

3.Evaluate best model:

python src/evaluate_best_model.py

4.Tune Gradient Boosting model:

python src/tune_model.py
