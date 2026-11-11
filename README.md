# Wine Quality Prediction System

An end-to-end multi-class classification pipeline that predicts wine quality scores (3–8) using 11 physicochemical properties from the `WineQT.csv` dataset.

---

## Project Overview

- **Goal:** Classify wine quality scores (3 to 8) based solely on chemical measurements.
- **Dataset:** `WineQT.csv` (1,143 raw records → 1,018 unique records after deduplication).
- **Best Performing Model:** Tuned Support Vector Machine (`SVC` with RBF kernel, `C=50`, `gamma=0.01`), achieving **60.3% test accuracy** and **60.9% 5-fold cross-validation accuracy**.

---

## Dataset & Features

The `Id` identifier column is dropped to prevent data leakage. The remaining 11 input features are:

| Feature | Description |
| :--- | :--- |
| `fixed acidity` | Non-volatile acids (mostly tartaric) contributing to sharpness |
| `volatile acidity` | Acetic acid content; high levels cause an unpleasant vinegar taste |
| `citric acid` | Adds freshness and flavor to wines |
| `residual sugar` | Remaining sugar after fermentation stops |
| `chlorides` | Salt content in the wine |
| `free sulfur dioxide` | Free form of $\text{SO}_2$ that prevents microbial growth and oxidation |
| `total sulfur dioxide` | Total amount of free and bound forms of $\text{SO}_2$ |
| `density` | Density of the wine, dependent on alcohol and sugar percentage |
| `pH` | Acidity scale from 0 (very acidic) to 14 (very basic) |
| `sulphates` | Wine additive contributing to $\text{SO}_2$ levels and antimicrobial protection |
| `alcohol` | Percent alcohol content by volume |

- **Target Variable:** `quality` (integer score between `3` and `8`).

---

## Workflow & Methodology

1. **Data Cleaning:**
   - Removed the non-predictive `Id` column.
   - Identified and dropped **125 exact duplicate rows** hidden by unique `Id` values to prevent train-test data leakage.
   - Verified zero missing or negative values; retained legitimate zero entries in `citric acid` and natural chemical outliers.
2. **Exploratory Data Analysis (EDA) & Correlation:**
   - Observed severe class imbalance (~83% of samples belong to quality `5` or `6`; quality `3` has only 6 records).
   - Identified `alcohol` ($r = +0.49$) and `volatile acidity` ($r = -0.41$) as the strongest linear correlates with wine quality.
3. **Preprocessing & Splitting:**
   - Performed an 80/20 stratified train-test split (`random_state=42`).
   - Applied `StandardScaler` fit strictly on training data to prevent data leakage.
4. **Model Training & Hyperparameter Tuning:**
   - Evaluated Logistic Regression, Random Forest, RBF SVM, and K-Nearest Neighbors (KNN).
   - Tuned the top-performing SVM model using `GridSearchCV` inside a `Pipeline` (`StandardScaler` + `SVC`) with 5-fold `StratifiedKFold` cross-validation.

---

## Model Benchmarks

| Model | Accuracy (Unscaled) | Accuracy (Scaled) | Weighted F1 (Scaled) |
| :--- | :---: | :---: | :---: |
| **Tuned SVM (RBF)** | — | **0.6029** | **0.5830** |
| **Baseline SVM (RBF)** | 0.5245 | 0.6029 | 0.5810 |
| **Logistic Regression** | 0.5637 | 0.5980 | 0.5764 |
| **Random Forest** | 0.5784 | 0.5686 | 0.5567 |
| **KNN ($k=5$)** | 0.4706 | 0.5588 | 0.5504 |

### Key Findings
- **Feature Importance:** `alcohol` is the #1 predictor across both Random Forest importance and linear correlation, followed by `sulphates`, `total sulfur dioxide`, and `volatile acidity`.
- **Impact of Scaling:** Feature scaling boosted SVM accuracy by **+7.8%** and KNN accuracy by **+8.8%**, while tree-based Random Forest remained unaffected.
- **Error Analysis:** **91% of all misclassifications** occur between adjacent quality tiers (e.g., `5` vs. `6`, or `7` vs. `6`), reflecting overlapping chemical profiles and subjective human tasting scores. Rare classes (`3`, `4`, `8`) suffer from low recall due to sample scarcity.

---

## Quality Tier Mapping

To make predictions actionable for non-technical users (winemakers, buyers, retailers), raw numeric scores are mapped to readable tiers:

| Score | Quality Category |
| :---: | :--- |
| `3 – 4` | **Poor** |
| `5` | **Average** |
| `6` | **Good** |
| `7 – 8` | **Excellent** |

---

## Setup & Usage

### Prerequisites
Install the required Python packages:
```bash
pip install numpy pandas matplotlib scikit-learnRunning the Project
Place WineQT.csv in the same directory as wine_quality_prediction.py.

Run the script or execute it cell-by-cell in Jupyter / Google Colab:

Bash
python wine_quality_prediction.py
Making Predictions
Use predict_wine_quality() with raw chemical values (scaling is handled automatically by the pipeline):

Python
predict_wine_quality(
    fixed_acidity=7.4,
    volatile_acidity=0.50,
    citric_acid=0.25,
    residual_sugar=2.1,
    chlorides=0.075,
    free_sulfur_dioxide=15,
    total_sulfur_dioxide=45,
    density=0.9965,
    pH=3.30,
    sulphates=0.70,
    alcohol=11.5
)
Sample Output:

Plaintext
----------------------------------------
Predicted Wine Quality : 6
Quality Category       : Good
Prediction Confidence  : 58%
----------------------------------------

Hover over the top-right corner of the box above and click the **Copy** icon to grab the entire
