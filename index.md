# CYT180 — Lab 8: Data Preprocessing for Machine Learning 
**Weight:** 3% <br>
**Work Type:** Individual <br>
**Submission Format:** 

----

## Introduction
In Lab 7, you completed all essential preprocessing tasks needed to prepare a structured dataset for supervised machine learning. You inspected the dataset, cleaned inconsistencies, analyzed distributions, removed outliers, evaluated correlations, examined class balance, separated features from the target variable, and applied feature scaling. These steps ensured that your dataset is clean, consistent, and mathematically ready for modeling.
In this lab, you will move into the modeling and evaluation stage of the machine‑learning workflow. You will split your dataset into training and testing sets, build a baseline classification model, generate predictions, evaluate model performance using industry‑standard metrics, visualize diagnostic plots, and interpret the results in a cybersecurity context.
This phase is critical: a model is only as useful as its ability to generalize to new, unseen data. The steps in Lab 8 are designed to help you understand not only how to build a model, but also how to evaluate it responsibly and translate metrics into practical insights for cybersecurity analytics.

----

## Learning Objectives
By the end of this lab you will be able to:
- Perform a train–test split using stratification
- Train a baseline classification model (Logistic Regression)
- Generate predictions and probability outputs
- Compute essential evaluation metrics including accuracy, precision, recall, and F1‑score
- Construct a confusion matrix and ROC curve and interpret AUC
- Visualize key performance results
- Interpret model behavior in a cybersecurity context
- Produce a structured analytical report summarizing methodology, results, and insights
----

## Section 1 — Data Setup (Load Raw Data and Apply Minimal Preprocessing)

In Lab 7, you performed a full preprocessing workflow that included cleaning, outlier removal, correlation analysis, class balance inspection, and scaling the feature matrix. However, Lab 8 may be run in a new notebook, and variables such as `clean_df`, `X`, and `X_normalized` will not exist automatically. To make Lab 8 self‑contained and reproducible, we reload the raw dataset and repeat the minimum required preprocessing steps, so that the data is ready for modeling.

Section 1 re‑creates the essential outputs of Lab 7:

- Load the original CSV
- Remove outliers using the same IQR technique
- Re‑separate features (X) and target (y)
- Perform train–test split
- Scale/Normalize features properly for machine learning
  
```python
# 1. Load raw dataset
import pandas as pd
import numpy as np
df = pd.read_csv("diabetes.csv")

# 2. Remove outliers from the Insulin column (same method from Lab 7)
q1, q3 = np.percentile(df["Insulin"].dropna(), [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df[(df["Insulin"] >= lower) & (df["Insulin"] <= upper)]

# 3. Separate features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# 4. Train–test split (stratified)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Fit MinMaxScaler on TRAIN ONLY (best practice)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train
X_test_scaled  = scaler.transform(X_test)       # transform test
```

You may be thinking why we split first, then scale (even though we scaled in Lab 7). In Lab 7, scaling was performed as part of exploratory data analysis, not model training. Its purpose was:
- to compare feature ranges
- to understand relationships
- to visually inspect normalized distributions
- to prepare for future modeling
- to introduce students to scaling early

However, the scaling done in Lab 7 is not appropriate for actual model training because scaling must always be fit on the training set only. If we scale **before** splitting, this happens:
- the scaler “looks at” the entire dataset, including test data.
- this gives the model information about the test set it shouldn’t have.
- this is called **data leakage**, and it makes evaluation unreliable  

Therefore, Lab 8 uses the correct machine‑learning practice:

```text
Split → Fit scaler on X_train → Transform X_train & X_test
```

This ensures that the model never sees test‑set information during training and that scaling parameters (min, max, mean, std) come only from the training data. This keeps evaluation metrics valid and trustworthy.
This is critically important in cybersecurity, where even small leaks can result in models that appear extremely accurate but fail catastrophically on real‑world threat data.
Now that the data is properly prepared, we can proceed to **Step 2: Train a Classification Model**.

----

## Section 2 — Train–Test Split
Before we train any machine‑learning model, we must split our dataset into two parts:

- Training set — used to fit/learn the model
- Testing set — used to evaluate how well the model generalizes to unseen data

This is one of the most important steps in the entire machine‑learning workflow.
A model that is trained and tested on the same data will appear artificially strong but will fail in real‑world situations.
Train–test splitting protects us against this by ensuring we evaluate the model only on data it has never seen before.


## Section 1 — Train–Test Split
Before we train any machine‑learning model, we must split our dataset into two parts:

- Training set — used to fit/learn the model
- Testing set — used to evaluate how well the model generalizes to unseen data

This is one of the most important steps in the entire machine‑learning workflow.
A model that is trained and tested on the same data will appear artificially strong but will fail in real‑world situations.
Train–test splitting protects us against this by ensuring we evaluate the model only on data it has never seen before.
