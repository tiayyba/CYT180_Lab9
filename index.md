# CYT180 — Lab 9: Machine‑Learning Model Training and Performance Evaluation
**Weight:** 3% <br>
**Work Type:** Individual <br>
**Submission Format:** 

----

## Introduction
In Lab 8, you completed all essential preprocessing tasks needed to prepare a structured dataset for supervised machine learning. You inspected the dataset, cleaned inconsistencies, analyzed distributions, removed outliers, evaluated correlations, examined class balance, separated features from the target variable, and applied feature scaling. These steps ensured that your dataset is clean, consistent, and mathematically ready for modeling.
In this lab, you will move into the modeling and evaluation stage of the machine‑learning workflow. You will split your dataset into training and testing sets, build a baseline classification model, generate predictions, evaluate model performance using industry‑standard metrics, visualize diagnostic plots, and interpret the results in a cybersecurity context.
This phase is critical: a model is only as useful as its ability to generalize to new, unseen data. The steps in Lab 9 are designed to help you understand not only how to build a model, but also how to evaluate it responsibly and translate metrics into practical insights for cybersecurity analytics.

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

In Lab 8, you performed a full preprocessing workflow that included cleaning, outlier removal, correlation analysis, class balance inspection, and scaling the feature matrix. Lab 9 is the continuation of Lab 8. To make Lab 8 self‑contained and reproducible, we reload the raw dataset and repeat the minimum required preprocessing steps, so that the data is ready for modeling and the variables such as `clean_df`, `X`, and `X_normalized` exist for further processing.

Section 1 re‑creates the essential outputs of Lab 8:

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

You may be thinking why we split first, then scale (even though in Lab 8 we scaled the date before spliting). In Lab 8, scaling was performed as part of exploratory data analysis, not model training. Its purpose was:
- to compare feature ranges
- to understand relationships
- to visually inspect normalized distributions
- to prepare for future modeling
- to introduce students to scaling early

However, the scaling done in Lab 8 is not appropriate for actual model training because scaling must always be fit on the training set only. If we scale **before** splitting, the following situations are possible:
- the scaler “looks at” the entire dataset, including test data.
- this gives the model information about the test set it shouldn’t have.
- this is called **data leakage**, and it makes evaluation unreliable  

Therefore, Lab 9 uses the correct machine‑learning practice:

```text
Split → Fit scaler on X_train → Transform X_train & X_test
```

This ensures that the model never sees test‑set information during training and that scaling parameters (min, max, mean, std) come only from the training data. This keeps evaluation metrics valid and trustworthy.
This is critically important in cybersecurity, where even small leaks can result in models that appear extremely accurate but fail catastrophically on real‑world threat data.
Now that the data is properly prepared, we can proceed to **Step 2: Train a Classification Model**.

----

## Step 2 — Train a Classification Model

Now that the dataset has been properly prepared (loaded, cleaned, split into training/testing sets, and scaled), we can train our first supervised machine‑learning model. In this lab, we will use **Logistic Regression**, a widely used baseline classifier for binary classification tasks. It is simple, fast, easy to interpret, and performs well when features have been normalized — which we ensured in Section 1.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```

### Explanation

- **Logistic Regression** learns a weight (coefficient) for each feature.
- During training, it identifies patterns in `X_train_scaled` that help distinguish class **0 (No Diabetes)** from class **1 (Diabetes)**.
- The model outputs a probability between 0 and 1 for each test sample, representing how likely the model believes the instance belongs to class 1.
- This model is commonly used in cybersecurity for tasks such as:
  - intrusion detection  
  - malware classification  
  - authentication anomaly detection  
  - fraud detection  

Logistic Regression serves as an excellent **baseline model** because:
- it is interpretable (weights show feature influence)
- it trains quickly
- it provides probability scores
- it highlights whether preprocessing was done correctly

You will later compare other models to this baseline in future labs.

### Reflection Questions

- Why is Logistic Regression considered a good baseline model for binary classification?
- What might happen if the features were **not** scaled before training?
- Why are we applying `fit_transform()` on Training data and `transform()` on test data?

----

## Section 3 — Generate Model Predictions

Once the model has been trained using the training data, the next step is to use it to generate predictions on the **test set**. This allows us to measure how well the model generalizes to new, unseen data. We will produce two types of predictions:

1. **Class labels** (0 or 1)  
2. **Probability scores** (between 0 and 1)

Probability predictions are especially important in cybersecurity settings where decisions often depend on risk thresholds rather than strict binary output.

```python
# Generate class label predictions
y_pred = model.predict(X_test_scaled)

# Generate probability predictions for the positive class (class 1)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

### Explanation

- `predict()`  
  Returns **hard labels** (0 = No Diabetes, 1 = Diabetes).  
  These are useful for confusion matrices and calculating metrics like accuracy.

- `predict_proba()`  
  Returns **probabilities**, not just labels.  
  The model outputs two probabilities per sample:
  - probability of class 0  
  - probability of class 1  

  By selecting `[:, 1]`, we extract the probability of belonging to class **1 (Diabetes)**.

Probability outputs are essential for:
- ROC curve  
- AUC calculation  
- threshold tuning  
- risk‑based decision making (very important in cybersecurity)

### Reflection Questions
- In threat detection, when might it be useful to adjust the classification threshold rather than rely on the default 0.5?
- How could probability outputs help in prioritizing cybersecurity alerts?

----

## Section 4 — Evaluation Metrics

After generating predictions on the test set, the next step is to evaluate how well the model performed.  
In cybersecurity, evaluation is especially important because different metrics reveal different types of model weaknesses — such as missed threats or excessive false alarms.

In this step, we compute five essential metrics:

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1‑score**  
- **Confusion Matrix**  
- **ROC Curve and AUC**

These metrics provide a complete picture of model performance and allow us to understand how well the model distinguishes between the two classes.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Classification metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
```

### Explanation of Metrics

**Accuracy**  
- Percentage of correctly predicted instances.  
- Can be misleading when classes are imbalanced.

**Precision**  
- Of the samples predicted as positive (class 1), how many were actually positive?  
- High precision = fewer false alarms.  
- Crucial when false positives waste analyst time (e.g., in intrusion detection systems).

**Recall**  
- Of all actual positives, how many did the model correctly detect?  
- High recall = fewer missed threats.  
- Extremely important in cybersecurity, where missing an attack can be costly.

**F1‑Score**  
- Harmonic mean of precision and recall.  
- Useful when both false positives and false negatives matter.

**Confusion Matrix**  
Shows the counts of:
- **True Positives (TP)**  
- **True Negatives (TN)**  
- **False Positives (FP)**  
- **False Negatives (FN)**  

A central tool for understanding model behavior.

**ROC Curve & AUC**  
- ROC shows the trade‑off between TPR (recall) and FPR at different thresholds.  
- **AUC** summarizes the model’s ability to distinguish between classes.  
  - **AUC ≈ 1.0** → Excellent separation  
  - **AUC ≈ 0.5** → Random guessing  

### Reflection Questions

- Why can accuracy be misleading in cybersecurity datasets with class imbalance?  
- For malware detection, which is more important — precision or recall — and why?  
- What information does the ROC curve provide that accuracy cannot?

----

## Section 5 — Visualizing Evaluation Metrics

Numerical evaluation metrics (accuracy, precision, recall, F1, AUC) are extremely valuable, but visualizations help us understand *how* the model behaves across different thresholds and where its mistakes occur. Visual tools make it easier to communicate model performance to both technical and non‑technical audiences — especially important in cybersecurity operations.

In this step, we visualize:

- **ROC Curve** (model performance across all thresholds)
- **Confusion Matrix** (counts of TP, TN, FP, FN)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# ROC curve plot
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # random baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

### Explanation

**ROC Curve**
- Shows how the model performs at **every possible classification threshold**.
- Each point represents a (False Positive Rate, True Positive Rate) pair.
- The diagonal line is random guessing.
- The farther the ROC curve is above that line, the better the model.
- The **AUC** (Area Under the Curve) summarizes overall performance.

**Confusion Matrix**
- Visualizes how many samples were correctly and incorrectly classified.
- Helps identify whether the model struggles more with:
  - **False Positives** (false alarms)
  - **False Negatives** (missed threats)
- In cybersecurity, this distinction is critical:
  - Too many FP → alert fatigue  
  - Too many FN → undetected attacks  

Combined, these visualizations give a deeper understanding of the model’s strengths and weaknesses beyond numerical metrics alone.

### Reflection Questions

- What does each point on the ROC curve represent in terms of model behavior?
- Why is the confusion matrix important even when accuracy, precision, and recall are already known?
- In cybersecurity, what real‑world situation corresponds to a **false positive**, and why can too many of them be problematic?

---- 

## Section 6 — Cybersecurity‑Focused Interpretation

Building a machine‑learning model is only the beginning.  
The most important step is **interpreting what the results mean**, especially in real‑world cybersecurity environments where decisions can affect detection accuracy, analyst workload, and risk levels.

In this step, you connect your evaluation metrics to operational meaning.  
Your goal is to understand **how the model behaves**, **what its strengths and weaknesses are**, and **how a SOC analyst would experience these outcomes**.

### Key Interpretation Areas

**1. Precision and False Alarms (False Positives)**  
Low precision means many benign events are flagged as malicious.  
In a SOC environment, this leads to:
- alert fatigue  
- wasted analyst time  
- delayed response to real incidents  

Too many false positives can make a detection model unusable.

---

**2. Recall and Missed Attacks (False Negatives)**  
Low recall means the model is missing actual threats.  
In cybersecurity, false negatives are often more dangerous than false positives because:
- intrusions go undetected  
- attackers dwell longer  
- incidents escalate before being caught  

A high-recall model is critical when the cost of missing an attack is severe.

---

**3. Confusion Matrix Insights**  
The confusion matrix reveals where the model struggles:
- High FN? → Model is too conservative, missing threats  
- High FP? → Model is too sensitive, generating noise  
- Balanced errors? → Threshold tuning may improve performance  

This matrix provides insight that accuracy alone cannot.

---

**4. AUC and Overall Separability**  
A high AUC value indicates the model can distinguish malicious vs. benign behavior across many thresholds — important in cybersecurity where thresholds are often adjusted dynamically.

AUC expresses overall model quality independent of a fixed cutoff.

---

**5. Feature Quality (Link to Lab 7)**  
The model’s performance reflects the quality of preprocessing done earlier:
- Were outliers handled correctly?  
- Did normalization improve consistency?  
- Did relevant features contribute strongly?  
- Did correlation analysis help avoid redundancy?  

Good preprocessing → more stable model.

---

### Reflection Questions

- In a SOC environment, would you prioritize precision or recall? Why?
- If the model has many false positives, how would this impact analyst workload?
- How might an attacker benefit from being misclassified due to low recall?
- How could scaling errors or poor preprocessing from Lab 7 distort interpretation in Lab 8?
- What additional cybersecurity features (e.g., login time, device ID, IP reputation) could improve the model’s performance?

----

## Section 7 — Reporting Template (Student Submission)

The final step of this lab is to create a short, structured report that summarizes everything you completed in Lab 8.  
This report should be **1–2 pages** and written in clear, professional language.  
Use the template below as your guide.  
You may include screenshots from your notebook where appropriate.

---

### **1. Introduction**
Briefly summarize:
- The purpose of Lab 8  
- What dataset was used  
- What problem the model is trying to solve  
- How this lab connects to the preprocessing work completed in Lab 7  

A good introduction answers:  
“What is this experiment about, and why does it matter?”

---

### **2. Methods**
Describe the steps you performed:
- How you reloaded and minimally preprocessed the dataset  
- How the data was split into training and testing sets  
- Why scaling was applied *after* splitting  
- What model you chose (Logistic Regression) and why it is an appropriate baseline  

This section explains *how* you conducted the experiment.

---

### **3. Results**
Include the key outputs from your evaluation:
- Accuracy  
- Precision  
- Recall  
- F1‑score  
- Confusion matrix  
- ROC curve with AUC value  

Use tables or screenshots to present results clearly.  
Focus on what the numbers show about the model’s behavior.

---

### **4. Interpretation**
Discuss the meaning of your results:
- What the confusion matrix tells you about false positives and false negatives  
- Whether precision or recall mattered more in this experiment  
- How well the ROC curve indicates class separability  
- How preprocessing affected model performance  
- How these results relate to cybersecurity use‑cases  

This is the most important part — explain what the results *mean*, not just what they *are*.

---

### **5. Conclusion**
Summarize your insights:
- Overall performance of the model  
- Strengths and weaknesses  
- Any limitations caused by the dataset  
- Recommendations for improving detection performance  
  (e.g., additional features, different models, threshold adjustments)

Keep this section concise and focused on big‑picture takeaways.

---

### **Report Submission Notes**
Your report should be:
- 1–2 pages  
- Professional and clearly organized  
- Written in complete sentences  
- Supported with screenshots or tables where helpful  

Make sure your writing focuses on both **technical correctness** *and* **cybersecurity relevance**.
