# -*- coding: utf-8 -*-
# %% [markdown]
# # Wine Quality Prediction System
# **End-to-End Machine Learning Project (Multi-class Classification)**
#
# Goal: predict the quality score (3 to 8) of a wine using only its chemical properties.
# Dataset: `WineQT.csv`
#
# Note to self: Id is just a row label, so it must never go into the model. Quality is the target.

# %% [markdown]
# ## TASK 1: Load and understand the dataset

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

pd.set_option("display.max_columns", 20)
np.random.seed(42)

# %%
# If you are on Colab, run this cell to upload WineQT.csv. If the file is already
# in the same folder, it just reads it directly.
import os
if not os.path.exists("WineQT.csv"):
    try:
        from google.colab import files
        uploaded = files.upload()      # pick WineQT.csv from your computer
    except ImportError:
        print("WineQT.csv not found, please put it next to this notebook")

df = pd.read_csv("WineQT.csv")
print("Loaded successfully")

# %%
print("First 5 records")
display(df.head()) if "display" in dir() else print(df.head())

# %%
print("Last 5 records")
print(df.tail())

# %%
print("5 random records")
print(df.sample(5, random_state=1))

# %%
print("Rows:", df.shape[0], "| Columns:", df.shape[1])
print("Target column  : quality")
print("Dropped later  : Id (just an identifier)")

# %% [markdown]
# What each feature means (quick notes for myself):
# - **fixed acidity**: non-volatile acids (mostly tartaric), gives the wine its sharpness
# - **volatile acidity**: acetic acid, too much gives a vinegar smell
# - **citric acid**: adds freshness
# - **residual sugar**: sugar left after fermentation
# - **chlorides**: salt content
# - **free / total sulfur dioxide**: preservative (SO2), protects against bacteria and oxidation
# - **density**: depends on alcohol and sugar
# - **pH**: acidity scale (lower is more acidic)
# - **sulphates**: additive that also acts as antimicrobial and antioxidant
# - **alcohol**: % by volume

# %% [markdown]
# ## TASK 2: Basic data inspection

# %%
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))

# %%
print(df.dtypes)

# %%
print(df.nunique())

# %%
print("Quality distribution (count)")
print(df["quality"].value_counts().sort_index())
print("\nQuality distribution (%)")
print((df["quality"].value_counts(normalize=True).sort_index() * 100).round(2))

# %%
print(df.describe().T)

# %%
print("Missing values per column:")
print(df.isnull().sum())
print("\nTotal missing:", df.isnull().sum().sum())
print("Duplicate rows (all columns incl. Id):", df.duplicated().sum())
print("Duplicate rows (ignoring Id)         :", df.drop(columns="Id").duplicated().sum())

# %% [markdown]
# ## TASK 3: Data cleaning

# %%
# Id is not a chemical property so it goes away
df_clean = df.drop(columns=["Id"])
print("Columns now:", list(df_clean.columns))

# %%
# missing values
if df_clean.isnull().sum().sum() == 0:
    print("No missing values, nothing to impute.")
else:
    df_clean = df_clean.fillna(df_clean.median())
    print("Filled missing values with median")

# %%
# duplicates: Id was hiding them, once Id is gone many rows are exact copies.
# If a copy lands in the test set while its twin is in train, the test score gets
# inflated (data leakage). So I remove them.
before = df_clean.shape[0]
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
print(f"Rows before: {before}, after removing duplicates: {df_clean.shape[0]}")
print(f"Removed {before - df_clean.shape[0]} duplicate rows")

# %%
# unusual / invalid values
print("Negative values in any column?", (df_clean < 0).any().any())
print("Zero values per column:")
print((df_clean == 0).sum()[lambda s: s > 0])   # citric acid can be 0 legitimately

print("\nAll columns numeric?", all(pd.api.types.is_numeric_dtype(t) for t in df_clean.dtypes))
df_clean = df_clean.astype(float).assign(quality=df_clean["quality"].astype(int))
print(df_clean.dtypes)

# %% [markdown]
# Notes: no negatives, nothing non-numeric. Zeros only appear in citric acid, which is
# possible (some wines just have none). I'll keep them. Extreme values do exist
# (e.g. chlorides, sulfur dioxide, residual sugar) but they look like real wines, not typos,
# so I'm not deleting them.

# %% [markdown]
# ## TASK 4: Exploratory Data Analysis

# %%
counts = df_clean["quality"].value_counts().sort_index()
plt.figure(figsize=(7, 4))
plt.bar(counts.index.astype(str), counts.values, color="#7b2d43")
for i, v in enumerate(counts.values):
    plt.text(i, v + 3, str(v), ha="center")
plt.title("Distribution of Wine Quality")
plt.xlabel("Quality score")
plt.ylabel("Number of wines")
plt.show()

print("Most common  :", counts.idxmax(), "->", counts.max(), "wines")
print("Least common :", counts.idxmin(), "->", counts.min(), "wines")

# %%
features = [c for c in df_clean.columns if c != "quality"]

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.ravel(), features):
    ax.hist(df_clean[col], bins=25, color="#c27c8f", edgecolor="white")
    ax.set_title(col)
for ax in axes.ravel()[len(features):]:
    ax.axis("off")
plt.suptitle("Feature distributions", fontsize=15)
plt.tight_layout()
plt.show()

# %%
# relationship of key features with quality (boxplots per quality level)
key_feats = ["alcohol", "volatile acidity", "sulphates", "pH", "citric acid"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.ravel(), key_feats):
    groups = [df_clean.loc[df_clean["quality"] == q, col] for q in sorted(df_clean["quality"].unique())]
    ax.boxplot(groups)
    ax.set_xticklabels(sorted(df_clean["quality"].unique()))
    ax.set_title(f"{col} vs quality")
    ax.set_xlabel("quality")
    ax.set_ylabel(col)
axes.ravel()[-1].axis("off")
plt.tight_layout()
plt.show()

# %%
print("Average value of key features for each quality level:")
print(df_clean.groupby("quality")[key_feats].mean().round(3))

# %%
# outliers using the IQR rule
outlier_counts = {}
for col in features:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_counts[col] = int(((df_clean[col] < low) | (df_clean[col] > high)).sum())

outlier_df = pd.Series(outlier_counts).sort_values(ascending=False)
print("Possible outliers per feature (IQR rule):")
print(outlier_df)

plt.figure(figsize=(9, 4))
outlier_df.plot(kind="bar", color="#7b2d43")
plt.title("Number of possible outliers per feature")
plt.ylabel("count")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## TASK 5: Correlation analysis

# %%
corr = df_clean.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

# %%
print("Correlation with quality (sorted by strength):")
q_corr = corr["quality"].drop("quality")
print(q_corr.reindex(q_corr.abs().sort_values(ascending=False).index).round(3))

# %%
print("Feature pairs with |correlation| > 0.6:")
found = False
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        r = corr.loc[features[i], features[j]]
        if abs(r) > 0.6:
            print(f"  {features[i]} <-> {features[j]} : {r:.2f}")
            found = True
if not found:
    print("  none")

# %% [markdown]
# Comparing with what I saw in EDA: it matches. In the boxplots / group means, the higher
# quality wines had clearly higher alcohol (avg ~9.9 for quality 5 vs ~11.5 for quality 7),
# lower volatile acidity (0.59 vs 0.39) and slightly higher sulphates and citric acid. The
# correlation numbers say the same thing: alcohol (+0.49) and volatile acidity (-0.41) are the
# strongest, then sulphates (+0.26) and citric acid (+0.24). pH, residual sugar and free
# sulfur dioxide are almost flat (|r| < 0.08).
# Strongly related to each other: fixed acidity with density (0.68), pH (-0.69) and citric
# acid (0.67), and free vs total sulfur dioxide (0.66). That makes sense chemically.

# %% [markdown]
# Is correlation alone enough to pick features? **No.** Correlation (Pearson) only captures
# *linear* relationships between two variables at a time. A feature can have a weak linear
# correlation with quality but still help when combined with other features, or when its
# effect is non-linear (e.g. sulphates only matter above a point). Also, two features that
# are highly correlated with each other (like fixed acidity, density, citric acid and pH)
# carry overlapping information. So I use correlation as a guide, not a final filter, and
# I let the models + feature importance tell the rest.

# %% [markdown]
# ## TASK 6: Feature and target separation

# %%
X = df_clean.drop(columns=["quality"])
y = df_clean["quality"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Features used:", list(X.columns))
print("Id in X?  ", "Id" in X.columns)
print("quality in X?", "quality" in X.columns)

# %% [markdown]
# Why the target must not be in the input features: if `quality` is given to the model as an
# input it can just copy it to produce the answer. It would show ~100% accuracy while
# learning nothing about wine chemistry, and it would fail on a new wine, because in real
# life we don't know the quality before predicting it. That's called target leakage.

# %% [markdown]
# ## TASK 7: Train-test split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])
dist = pd.DataFrame({
    "full %": (y.value_counts(normalize=True) * 100).round(2),
    "train %": (y_train.value_counts(normalize=True) * 100).round(2),
    "test %": (y_test.value_counts(normalize=True) * 100).round(2),
}).sort_index()
print(dist)

# %% [markdown]
# **Why not evaluate on the training data?** The model has already seen those rows, so a
# good score there only proves it can remember them. We want to know how it does on wines it
# has never seen, and that is what the test set is for.
#
# **Overfitting here:** the model learns tiny details/noise of the training wines (for
# example an exact combination of sulphates and chlorides) instead of general patterns.
# Train accuracy gets very high, test accuracy is much lower. Quality labels are given by
# human tasters, so they are noisy, which makes overfitting easy.

# %% [markdown]
# ## TASK 8: Feature scaling

# %%
print("Feature ranges before scaling:")
print(X_train.agg(["min", "max"]).T)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit only on train
X_test_scaled = scaler.transform(X_test)         # reuse the same mean/std for test

print("\nMean after scaling (train):", X_train_scaled.mean(axis=0).round(2))
print("Std  after scaling (train):", X_train_scaled.std(axis=0).round(2))

# %% [markdown]
# Total sulfur dioxide goes up to ~hundreds, while chlorides are around 0.05 to 0.1 and
# density is ~0.99 to 1.00. In **distance-based** models (KNN) the feature with the biggest
# numbers dominates the distance, so sulfur dioxide would basically decide everything. In
# **margin-based** models (SVM) the same thing happens with the margin, and logistic
# regression converges slower / regularisation hits features unevenly.
#
# - Sensitive to scaling: KNN, SVM, Logistic Regression
# - Generally not affected: Random Forest / Decision Trees (they only look at thresholds
#   per feature, so units do not matter)

# %% [markdown]
# ## TASK 9: Model training

# %%
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

trained = {}
preds = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained[name] = model
    preds[name] = model.predict(X_test_scaled)
    print(f"{name:20s} trained | test accuracy = {accuracy_score(y_test, preds[name]):.4f}")

# %% [markdown]
# Why each model suits (or not) this problem:
# - **Logistic Regression**: simple, fast baseline and gives probabilities. But it assumes
#   linear boundaries, and wine quality depends on feature interactions, so it can struggle.
# - **Random Forest**: handles non-linear relationships and interactions, robust to outliers
#   and scaling, and gives feature importance. Usually the best on this type of tabular data.
# - **SVM (RBF)**: can model non-linear boundaries, but needs scaling and is sensitive to
#   class imbalance and to C / gamma.
# - **KNN**: very intuitive (similar wines get similar scores) but suffers when classes
#   overlap and with 11 dimensions; it is also slower at prediction time.

# %% [markdown]
# ## TASK 10: Model evaluation

# %%
rows = []
for name, p in preds.items():
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, p, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, p, average="weighted", zero_division=0),
    })
results = pd.DataFrame(rows).set_index("Model").round(4)
results = results.sort_values("F1-score", ascending=False)
print(results)

best_name = results["Accuracy"].idxmax()
worst_name = results["Accuracy"].idxmin()
print("\nBest model  :", best_name)
print("Worst model :", worst_name)

# %%
# is the model equally good on every quality score?
print(f"Per-class report for {best_name}:\n")
print(classification_report(y_test, preds[best_name], zero_division=0))

# %% [markdown]
# Accuracy can be misleading because the classes are imbalanced: most wines are 5 or 6. A
# model that only predicts 5 and 6 can still score a decent accuracy while never getting a
# single 3, 4 or 8 right. That is why I also look at precision, recall and F1 per class
# (weighted average above, per-class in the report).

# %% [markdown]
# ## TASK 11: Confusion matrix analysis

# %%
best_model = trained[best_name]
labels = sorted(y.unique())
cm = confusion_matrix(y_test, preds[best_name], labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Reds", colorbar=False)
plt.title(f"Confusion matrix - {best_name}")
plt.show()

# %%
cm_df = pd.DataFrame(cm, index=[f"actual {l}" for l in labels], columns=[f"pred {l}" for l in labels])
print(cm_df)

print("\nPer-class recall (how many actual wines of this score were found):")
for i, l in enumerate(labels):
    tot = cm[i].sum()
    print(f"  quality {l}: {cm[i, i]}/{tot} = {cm[i, i] / tot:.2%}" if tot else f"  quality {l}: no samples")

# %%
# which pairs are confused the most (off-diagonal)
pairs = []
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        if i != j and cm[i, j] > 0:
            pairs.append((a, b, cm[i, j]))
pairs.sort(key=lambda t: -t[2])
print("Most frequent mistakes (actual -> predicted):")
for a, b, c in pairs[:6]:
    print(f"  {a} -> {b} : {c} times")

near = sum(c for a, b, c in pairs if abs(a - b) == 1)
total_err = sum(c for a, b, c in pairs)
print(f"\n{near} of {total_err} mistakes ({near / total_err:.0%}) are to a neighbouring quality level")

# %% [markdown]
# Why 5 gets predicted as 6 (and 7 as 6): quality is a human taster's score, and the
# difference between a 5 and a 6 is tiny and subjective. The chemical profiles of these
# neighbouring classes overlap heavily, so there is no clean boundary. Class 6 is also large,
# so the model leans towards it when unsure. Yes, the model clearly struggles to separate
# neighbouring levels, while its mistakes are mostly "off by one" rather than wild.

# %% [markdown]
# ## TASK 12: Feature importance analysis

# %%
rf = trained["Random Forest"]   # tree-based model
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature ranking (Random Forest):")
for rank, (f, v) in enumerate(importance.items(), 1):
    print(f"{rank:2d}. {f:22s} {v:.4f}")

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh", color="#7b2d43")
plt.title("Feature importance - Random Forest")
plt.xlabel("importance")
plt.tight_layout()
plt.show()

print("\nMost important :", importance.index[0])
print("Least important:", list(importance.index[-3:]))

# %%
compare = pd.DataFrame({
    "importance": importance,
    "abs corr with quality": q_corr.abs(),
}).sort_values("importance", ascending=False).round(3)
print(compare)

# %% [markdown]
# A feature with low correlation can still help a model because correlation only measures a
# straight-line relationship with the target alone. A tree can use that feature in
# combination with another one (for example sulphates matter more when alcohol is low), or
# in a non-linear / threshold way (only very high values matter). So low correlation does
# not mean useless.

# %% [markdown]
# ## TASK 13: Model improvement
# SVM was the best model in Task 10, so that is the one I try to improve.
# The main knobs of an RBF SVM are `C` (how strict the model is about training mistakes)
# and `gamma` (how far the influence of one training wine reaches).

# %%
from sklearn.pipeline import Pipeline

# trying a few manual settings (data is already scaled, so I use X_train_scaled)
experiments = {
    "baseline (C=1, gamma=scale)": dict(C=1, gamma="scale"),
    "C=0.5": dict(C=0.5),
    "C=5": dict(C=5),
    "C=20": dict(C=20),
    "gamma=0.01": dict(C=1, gamma=0.01),
    "gamma=0.1": dict(C=1, gamma=0.1),
    "C=10, gamma=0.1": dict(C=10, gamma=0.1),
    "class_weight=balanced": dict(C=1, class_weight="balanced"),
    "linear kernel": dict(kernel="linear", C=1),
}

rows = []
for name, params in experiments.items():
    m = SVC(random_state=42, **params).fit(X_train_scaled, y_train)
    rows.append({
        "Setting": name,
        "Train acc": accuracy_score(y_train, m.predict(X_train_scaled)),
        "Test acc": accuracy_score(y_test, m.predict(X_test_scaled)),
    })
exp_df = pd.DataFrame(rows).set_index("Setting").round(4)
exp_df["Gap"] = (exp_df["Train acc"] - exp_df["Test acc"]).round(4)
print(exp_df)

# %% [markdown]
# Looking at the table: a bigger C or gamma pushes the training accuracy up (the model
# bends itself around the training wines) but the test accuracy does not follow. So a higher
# training score does **not** mean a better model. What matters is the test /
# cross-validated score and a small train-test gap.

# %% [markdown]
# ## TASK 14: Hyperparameter tuning (GridSearchCV)

# %%
# Pipeline = scaler + SVM. Putting the scaler inside means that during cross-validation
# the scaler is fitted only on the training folds (no leakage from the validation fold).
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(random_state=42)),
])

param_grid = {
    "svc__C": [0.1, 0.5, 1, 2, 5, 10, 50],
    "svc__gamma": ["scale", 0.005, 0.01, 0.05, 0.1, 0.5],
    "svc__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best CV accuracy:", round(grid.best_score_, 4))

# %%
# original (default) SVM as a pipeline, for a fair comparison
orig_pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(random_state=42))])
orig_cv = cross_val_score(orig_pipe, X_train, y_train, cv=cv).mean()
orig_pipe.fit(X_train, y_train)
orig_pred = orig_pipe.predict(X_test)
orig_acc = accuracy_score(y_test, orig_pred)

# tuned model: same pipeline with the best params, probability=True so that we can
# show the confidence in the prediction system
best_svc = {k.replace("svc__", ""): v for k, v in grid.best_params_.items()}
tuned = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=42, **best_svc)),
])
tuned.fit(X_train, y_train)
tuned_pred = tuned.predict(X_test)
tuned_acc = accuracy_score(y_test, tuned_pred)

final_cmp = pd.DataFrame({
    "CV accuracy": [orig_cv, grid.best_score_],
    "Test accuracy": [orig_acc, tuned_acc],
    "Test F1 (weighted)": [f1_score(y_test, orig_pred, average="weighted", zero_division=0),
                           f1_score(y_test, tuned_pred, average="weighted", zero_division=0)],
}, index=["Original SVM", "Tuned SVM"]).round(4)
print(final_cmp)

# %% [markdown]
# Why tuning matters for a production model: default settings are rarely the best for a
# given dataset. Tuning with cross-validation picks settings based on several train/validation
# splits rather than one lucky split, which gives a more trustworthy and stable model and
# helps control overfitting. The test set is used only once, at the end.

# %% [markdown]
# ## TASK 15: Wine quality prediction system

# %%
# Final model: tuned SVM pipeline (StandardScaler + SVM). The scaler is inside the
# pipeline, so a new wine can be given in its normal raw units.
final_model = tuned

feature_order = list(X.columns)

def quality_category(score):
    """Task 16: turn the numeric score into a readable category"""
    if score <= 4:
        return "Poor"
    elif score == 5:
        return "Average"
    elif score == 6:
        return "Good"
    else:
        return "Excellent"

def predict_wine_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                         density, pH, sulphates, alcohol):
    new_wine = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                              chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                              density, pH, sulphates, alcohol]], columns=feature_order)
    pred = int(final_model.predict(new_wine)[0])
    proba = final_model.predict_proba(new_wine)[0]
    confidence = proba[list(final_model.classes_).index(pred)] * 100

    print("-" * 40)
    print(f"Predicted Wine Quality : {pred}")
    print(f"Quality Category       : {quality_category(pred)}")
    print(f"Prediction Confidence  : {confidence:.0f}%")
    print("-" * 40)
    return pred, quality_category(pred), confidence

# %%
# testing on a sample wine
predict_wine_quality(
    fixed_acidity=7.4, volatile_acidity=0.50, citric_acid=0.25, residual_sugar=2.1,
    chlorides=0.075, free_sulfur_dioxide=15, total_sulfur_dioxide=45,
    density=0.9965, pH=3.30, sulphates=0.70, alcohol=11.5
)

# %%
# another one with low alcohol / high volatile acidity
predict_wine_quality(
    fixed_acidity=7.8, volatile_acidity=0.88, citric_acid=0.0, residual_sugar=2.6,
    chlorides=0.098, free_sulfur_dioxide=25, total_sulfur_dioxide=67,
    density=0.9968, pH=3.20, sulphates=0.55, alcohol=9.4
)

# %%
# interactive version (type your own values)
def predict_from_input():
    names = ["Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", "Chlorides",
             "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density", "pH", "Sulphates", "Alcohol"]
    values = [float(input(f"{n}: ")) for n in names]
    return predict_wine_quality(*values)

# uncomment to try it
# predict_from_input()

# %% [markdown]
# ## TASK 16: Quality interpretation
#
# | Score | Category |
# |-------|----------|
# | 3-4   | Poor |
# | 5     | Average |
# | 6     | Good |
# | 7-8   | Excellent |
#
# A non-technical person (a shop owner, a buyer, a winemaker) does not know if "6" is good or
# bad. "Good" or "Excellent" is instantly clear and can directly drive decisions such as
# pricing or whether a batch needs a second look. It also hides the small 5-vs-6 noise a bit.

# %%
test_view = pd.DataFrame({"actual": y_test.values, "predicted": tuned_pred})
test_view["actual category"] = test_view["actual"].apply(quality_category)
test_view["predicted category"] = test_view["predicted"].apply(quality_category)
print(test_view.head(10))
print("\nAccuracy at category level:",
      round((test_view["actual category"] == test_view["predicted category"]).mean(), 4))

# %% [markdown]
# ## TASK 17: Final model analysis

# %%
# did scaling help? Same models with and without scaling
scale_rows = []
for name, mdl in {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}.items():
    raw = accuracy_score(y_test, mdl.fit(X_train, y_train).predict(X_test))
    sc = accuracy_score(y_test, mdl.fit(X_train_scaled, y_train).predict(X_test_scaled))
    scale_rows.append({"Model": name, "Without scaling": raw, "With scaling": sc, "Change": sc - raw})
scale_df = pd.DataFrame(scale_rows).set_index("Model").round(4)
print(scale_df)

# %%
print("Best algorithm      :", best_name)
print("Most important feat (RF):", importance.index[0])
print("Original SVM test acc:", round(orig_acc, 4), "| Tuned SVM test acc:", round(tuned_acc, 4))
print("\nPer-class report for the tuned model:")
print(classification_report(y_test, tuned_pred, zero_division=0))

# %% [markdown]
# ### Answers
# (see the printed outputs above, the summary below is based on the numbers from my run)
#
# **1. Most influential feature:** alcohol. It is #1 in Random Forest importance (~0.15) and
# also has the highest correlation with quality (+0.49). Next are sulphates, total sulfur
# dioxide and volatile acidity. The least important were fixed acidity, residual sugar and
# free sulfur dioxide, but none of them is really "zero", the importances are quite spread.
#
# **2. Best algorithm:** SVM (RBF) with ~60.3% test accuracy and weighted F1 ~0.58, closely
# followed by Logistic Regression (59.8%). Random Forest got 56.9% and KNN 55.9%.
# Note: the differences between the top models are small (a few wines out of 204), so with
# a test set this small I would not say SVM is dramatically better.
#
# **3. Why SVM did better:** after scaling, the RBF kernel can draw a smooth non-linear
# boundary but with a limited complexity (C=1), so it did not memorise the noisy labels.
# Random Forest memorised the training set (train acc = 100%) and only got ~57% on test,
# a classic overfitting case on a small and noisy dataset (~1000 rows after cleaning).
#
# **4. Did scaling help?** Yes for the scale-sensitive models: SVM 52.5% -> 60.3%, KNN
# 47.1% -> 55.9%, Logistic Regression 56.4% -> 59.8%. Random Forest did not need it (57.8%
# -> 56.9%, just random variation).
#
# **5. Did tuning help?** Only a little. The CV accuracy went from 0.593 to 0.609, but the
# final test accuracy stayed 60.3% (F1 0.581 -> 0.583). Best params were C=50, gamma=0.01.
# So tuning gave a slightly more reliable model, but not a big jump. The ceiling is probably
# the data rather than the settings. Also, bigger C/gamma made train accuracy go up to 85%
# while test accuracy dropped to ~52%, which again shows more training accuracy is not better.
#
# **6. Difficult classes:** Quality 3, 4 and 8 are basically never predicted correctly (0 recall),
# because they have only 6, ~33 and ~15 samples in total. Classes 6 and 7 are also hard:
# recall for 7 is only ~38%, and most of the confusion is 6<->5 and 7->6. About 91% of all
# mistakes are to the neighbouring quality level, so the model is "off by one" mostly.
#
# **7. Limitations of the dataset:**
# - Very imbalanced: ~83% of wines are quality 5 or 6, only 6 wines have quality 3.
# - Small dataset (1143 rows, 1018 after removing 125 duplicates).
# - The target is a subjective score from human tasters, so the labels are noisy.
# - Only chemical measurements. Grape type, region, vintage, brand etc. are not included.
# - Probably only one wine type (looks like red wine), so it may not generalise to other wines.
# - Accuracy ~60% means that this is a helpful assistant, not a replacement for tasters.

# %% [markdown]
# ## TASK 18: Final conclusion
#
# **Dataset understanding:** WineQT has 1143 wines with 11 chemical measurements, an Id and a
# quality score from 3 to 8. It is a multi-class classification problem and the classes are
# heavily imbalanced (most wines are 5 or 6).
#
# **Data cleaning:** No missing values and no invalid (negative / non-numeric) values.
# I dropped Id, and found that 125 rows were exact duplicates once Id was ignored, so I
# removed them to avoid leakage between train and test. 1018 rows were left.
#
# **EDA:** Quality 5 is the most common (433 wines), 3 the rarest (6 wines). Better wines
# tend to have more alcohol, less volatile acidity and more sulphates / citric acid. Some
# features (residual sugar, chlorides, sulphates) have a lot of outliers, but they looked
# like real wines so I kept them.
#
# **Feature engineering and scaling:** Used all 11 chemical features. Applied StandardScaler
# (fitted on training data only). Scaling helped SVM, KNN and Logistic Regression a lot
# and made no real difference for Random Forest.
#
# **Models used:** Logistic Regression, Random Forest, SVM and KNN.
#
# **Model evaluation:** Accuracy, precision, recall and weighted F1. SVM was the best
# (accuracy 60.3%, F1 0.58) and KNN the worst (55.9%, F1 0.55). Per-class results show
# that rare classes (3, 4, 8) are never found, so accuracy alone hides that.
#
# **Best model and tuning:** SVM. GridSearchCV with 5-fold CV (using a pipeline so that
# scaling is done inside each fold) found C=50, gamma=0.01, with a best CV score of 0.609.
# Test accuracy stayed at 60.3%, so tuning mostly made the result more trustworthy
# than better.
#
# **Prediction system:** `predict_wine_quality()` takes the 11 chemical values of a wine and
# returns the predicted score, the category (Poor / Average / Good / Excellent) and the
# confidence (probability of the predicted class).
#
# **Important findings:** alcohol and volatile acidity are the key drivers of quality;
# most errors are between neighbouring scores; high training accuracy (RF 100%) does not
# mean a good model; and the dataset itself (imbalance, noisy labels) limits the score.
#
# **Real-world applications:** quick quality screening of batches in wineries, helping
# decide which batches should go to expert tasting, price estimation, quality control
# monitoring during fermentation, and helping retailers/buyers compare wines.
#
# **What I learned:** the full ML workflow from cleaning to deployment-style prediction,
# why duplicates and data leakage matter, why scaling is needed for some algorithms,
# why accuracy alone is misleading on imbalanced data, how to read a confusion matrix, and
# how overfitting looks like in numbers (train 100% vs test ~57%). Also that a more complex
# model or more tuning does not automatically give a better result; better data
# (more samples for the rare classes) would probably help more.
