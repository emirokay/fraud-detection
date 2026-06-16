# Credit Card Fraud Detection

Detect fraudulent transactions from 28 PCA-transformed features using
Logistic Regression, Random Forest, and XGBoost.

## Problem

Credit card fraud accounts for billions in annual losses. With only 0.17%
of transactions being fraudulent, this is an extreme class imbalance problem.
The dataset (Kaggle) contains 284,807 European cardholder transactions with
PCA-transformed features for confidentiality.

## Approach

| Step | What |
|------|------|
| **EDA** | Class distribution, feature correlations, amount analysis |
| **Feature Engineering** | Cyclical time encoding (hour sin/cos), scaled Amount |
| **Modeling** | 3 models with default params + class imbalance handling |
| **Evaluation** | PR-AUC (precision-recall), confusion matrices, feature importance |

## Results

| Model | PR-AUC | Precision | Recall | F1 |
|-------|--------|-----------|--------|-----|
| XGBoost | **0.819** | 0.913 | 0.768 | 0.834 |
| Random Forest | 0.815 | **0.959** | 0.737 | 0.833 |
| Logistic Regression | 0.676 | 0.056 | 0.874 | 0.105 |

**Best for production:** Random Forest — near-identical PR-AUC to XGBoost
but with higher precision (0.959), meaning fewer false alarms for fraud teams.

**Key insight:** V14 is the strongest fraud predictor, followed by V17 and V10.
Tree-based models outperform logistic regression, confirming non-linear
patterns in the data.

## How to Run

1. Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and save as `data.csv`
2. `pip install -r requirements.txt`
3. `jupyter notebook fraud_detection.ipynb`

## Built With

pandas, scikit-learn, matplotlib, seaborn, XGBoost
