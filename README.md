# Thyroid Cancer Recurrence Prediction

Predict whether well-differentiated thyroid cancer will recur using clinicopathologic features and scikit-learn classification models.

## Problem

Thyroid cancer recurrence affects roughly 28% of patients in this cohort. The goal is to build models that flag high-risk patients early, using demographics, clinical history, tumor pathology, and TNM staging features.

## Dataset

- **Source:** [UCI ML Repository: Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)
- **Size:** 383 patients, 16 features, 15-year follow-up (minimum 10 years per patient)
- **Target:** `Recurred` (`Yes` / `No`)

## Approach

1. **Exploratory analysis** — class balance, recurrence rates by stage/risk/response, age distributions
2. **Preprocessing** — one-hot encoding for categoricals, standard scaling for age, all inside a scikit-learn `Pipeline`
3. **Model comparison** — Logistic Regression, Linear SVM, Random Forest, and SGD Classifier with 5-fold stratified cross-validation
4. **Two feature sets:**
   - **Baseline** — diagnosis-time features (no post-treatment `Response`)
   - **Extended** — includes `Response` to show how much performance increases with post-treatment information
5. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, ROC curve, and feature importance

## Results (high level)

| Setup | Best model | Approx. F1 |
|---|---|---|
| Baseline (no `Response`) | Random Forest | ~72% |
| Extended (with `Response`) | Logistic Regression | ~93% |

Including post-treatment response dramatically improves predictions, which highlights an important clinical modeling choice: features available at diagnosis vs. after therapy.

## Run the notebook

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook predicting_recurrence_of_tc.ipynb
```

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Project structure

```
.
├── predicting_recurrence_of_tc.ipynb   # Main analysis notebook
├── Thyroid_Diff.csv                    # Dataset
├── requirements.txt
└── README.md
```

## References

Borzooei, S. & Tarokhian, A. (2023). *Differentiated Thyroid Cancer Recurrence* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5632J](https://doi.org/10.24432/C5632J)
