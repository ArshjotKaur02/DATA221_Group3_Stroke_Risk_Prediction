"""
Stroke Prediction - Logistic Regression Model

This script trains a logistic Regression classifier on the preprocessed stroke
dataset. Because the dataset is heavily imbalanced (~4.9% positive stroke cases),
class_weight="balanced" is used so the model does not simply predict the majority
class every time.
"""

#------------------------------------ IMPORTS ------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay,)

#------------------------------------ Reproduce PreProcessing From DATA_PROCESSING.py ------------------------------------
# Copy the preprocessing pipeline so this file runs standalone
patient_records_dataframe = pd.read_csv("healthcare_stroke_data.csv")
patient_records_dataframe = patient_records_dataframe.drop(columns = ["id"])

# Handle missing BMI values
patient_records_dataframe["bmi"] = pd.to_numeric(patient_records_dataframe["bmi"]
, errors = "coerce")
median_bmi_value = patient_records_dataframe["bmi"].median()
patient_records_dataframe["bmi"] = patient_records_dataframe["bmi"].fillna(median_bmi_value)

# Binary encode Yes/No columns
patient_records_dataframe["ever_married"] = patient_records_dataframe["ever_married"].map({"Yes": 1, "No": 0})
patient_records_dataframe["Residence_type"] = patient_records_dataframe["Residence_type"].map({"Urban": 1, "Rural": 0})

# One-hot encode multi-category columns
patient_records_dataframe = pd.get_dummies(patient_records_dataframe, columns = ["gender", "work_type", "smoking_status"],
drop_first = True)

# Separate features and target
feature_matrix = patient_records_dataframe.drop(columns = ["stroke"])
target_label = patient_records_dataframe["stroke"]

# 70/30 stratified split (same seed as Data_Processing.py)
feature_train, feature_test, target_train, target_test = train_test_split(
    feature_matrix, target_label,
    test_size = 0.3, random_state = 42, stratify = target_label)

#------------------------------------ FEATURE SCALING ------------------------------------
"""
Logistic Regression uses gradient-based optimisation; features on very different scales
(e.g. age = 0-100 vs avg_glucose_level = 50-280) can slow convergence and distort the magnitude
of the learned coefficients. Standard centres each feature to mean = 0 and scales to unit variance.
IMPORTANT: Fit ONLY on the training set, then transform both sets. Fitting on the full dataset
would cause data leakage (the scaler would "know" about test-set statistics before the model has
seen the test data).
"""

scaler = StandardScaler()
feature_train_scaled = scaler.fit_transform(feature_train) # fit + transform on train
feature_test_scaled = scaler.transform(feature_test) # transform only on test

#------------------------------------ MODEL TRAINING ------------------------------------
"""
Key hyperparameter choices: 

class_weight = "balanced"
    The dataset has ~4.9% positive stroke cases (249 out of 5110). Without this, the model 
    would achieve ~95% accuracy by always predicting "no stroke". "balanced" automatically
    weights each class inversely proportional to its frequency, penalising misclassification
    of the minority (stroke = 1) class more heavily. 

solver = "lbfgs"
    A quasi-Newton optimiser that works well for small-to-medium datasets and supports the
    L2 penalty used by default.

max-iter = 1000
    The default of 100 iterations can fail to converge on this dataset after scaling; 1000
    gives the solver sufficient room to reach the optimum.

"""

logistic_model = LogisticRegression(
    class_weight = "balanced", solver = "lbfgs", max_iter = 1000, random_state = 42)
logistic_model.fit(feature_train_scaled, target_train)

#------------------------------------ PREDICTIONS ------------------------------------
target_predictor = logistic_model.predict(feature_test_scaled) # Hard class labels (0 or 1)
target_predictor_probability = logistic_model.predict_proba(feature_test_scaled)[:,1] # Probability of stroke = 1

#------------------------------------ EVALUATION ------------------------------------
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
"""
The classification report shows precision, recall, and F1 for each class. For an imbalanced
dataset, recall on class 1 (stroke detected) is the most clinically important metric. We want
to minimize false negatives.
"""
print(classification_report(target_test, target_predictor, target_names = ["No Stroke", "Stroke"]))
roc_auc = roc_auc_score(target_test, target_predictor_probability)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print()
"""
ROC-AUC measures the model's ability to discriminate between stroke and non-stroke
patients across ALL decision thresholds. It is unaffected by class imbalance
and ranges from 0.5 (random guessing) to 1.0 (perfect separation).
"""

#------------------------------------ CONFUSION MATRIX PLOT ------------------------------------
fig, axes = plt.subplots(1, 2, figsize = (14, 5))

# --- Left: Confusion Matrix ---
cm = confusion_matrix(target_test, target_predictor)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["No Stroke", "Stroke"])
disp.plot(ax = axes[0], colorbar = False, cmap = "Blues")
axes[0].set_title("Confusion Matrix", fontsize = 13, fontweight = "bold")

# Annotate each cell with a short label so the reader understands what TN/FP/FN/TP mean
cell_labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        axes[0].text(j, i-0.3, cell_labels[i][j], ha = "center", va = "center", fontsize = 9, color = "grey")

# --- Right: ROC Curve ---
fpr, tpr, _ = roc_curve(target_test, target_predictor_probability)
axes[1].plot(fpr, tpr, color = "steelblue", lw = 2, label = f"Logistic Regression (AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], color = "grey", linestyle = "--", lw = 1, label = "Random Classifier")
axes[1].fill_between(fpr, tpr, alpha = 0.08, color = "steelblue")
axes[1].set_xlabel("False Positive Rate", fontsize = 11)
axes[1].set_ylabel("True Positive Rate (Recall)", fontsize = 11)
axes[1].set_title("ROC Curve", fontsize = 13, fontweight = "bold")
axes[1].legend(loc = "lower right")
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])

plt.suptitle("Logistic Regression - Stroke Prediction Evaluation", fontsize = 14, fontweight = "bold", y = 1.02)
plt.tight_layout()
plt.savefig("stroke_model_evaluation.png", dpi = 150, bbox_inches = "tight")
plt.show()
print("Evaluation plot saved to stroke_model_evaluation.png")

#------------------------------------ FEATURE IMPORTANCE ------------------------------------
"""
Logistic Regression assigns a coefficient to every feature. After scaling, the magnitude
of a coefficient reflects how strongly that feature influences the log-odds of a stroke-positive
= increases risk, negative = decreases risk.
"""

coef_series = pd.Series(
    logistic_model.coef_[0], index = feature_matrix.columns,).sort_values(key = abs, ascending = False)
print("=" * 60)
print("TOP 10 FEATURES BY COEFFICIENT MAGNITUDE")
print("=" * 60)
print(coef_series.head(10).to_string())
print()

# Plot top 10 features
top10 = coef_series.head(10)
colors = ["#d73027" if v > 0 else "#4575b4" for v in top10.values]
plt.figure(figsize = (9, 5))
bars = plt.barh(top10.index[::-1], top10.values[::-1], color = colors[::-1])
plt.axvline(0, color = "black", linewidth = 0.8)
plt.xlabel("Coefficient Value (scaled)", fontsize = 11)
plt.title("Top 10 Feature Coefficients\n(red = increases stroke risk, blue = decreases)", fontsize = 12, fontweight = "bold")
plt.tight_layout()
plt.savefig("stroke_feature_importance.png", dpi = 150, bbox_inches = "tight")
plt.show()
print("Feature importance plot saved to stroke_feature_importnace.png")