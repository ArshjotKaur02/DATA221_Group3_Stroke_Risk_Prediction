"""
Stroke Prediction - XGBoost Model
Imports preprocessed splits directly from Stroke_Data_Processing.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)


# Import the already-prepared train/test splits from the preprocessing module
from Stroke_Data_Processing import (
    feature_train, feature_test,
    target_train, target_test,
    feature_matrix
)

neg_count = (target_train == 0).sum()
pos_count = (target_train == 1).sum()
scale_pos_weight = neg_count / pos_count


# Key hyperparameters explained:
#   n_estimators      – number of boosting rounds
#   learning_rate     – (eta) shrinks each tree's contribution
#   max_depth         – maximum depth of each tree
#   subsample         – fraction of rows sampled per tree (row subsampling)
#   colsample_bytree  – fraction of features sampled per tree (column subsampling)
#   min_child_weight  – minimum sum of instance weight in a leaf;
#                       higher values prevent overfitting on rare cases
#   gamma             – minimum loss reduction required to make a split;
#                       acts as a pruning threshold
#   reg_alpha         – L1 regularisation on leaf weights (sparsity)
#   reg_lambda        – L2 regularisation on leaf weights (smoothing)
#   scale_pos_weight  – balances positive/negative class weights natively
#   eval_metric       – use AUC so early stopping tracks the right metric
#   use_label_encoder – suppressed to avoid deprecation warnings

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=30,   # stop if AUC doesn't improve for 30 rounds
    random_state=17
)


# Passing eval_set lets XGBoost monitor validation AUC each round
# and stop early if it stops improving, preventing overfitting.

print("Training XGBoost model …")
xgb_model.fit(
    feature_train, target_train,
    eval_set=[(feature_test, target_test)],
    verbose=50      # print progress every 50 rounds
)

best_round = xgb_model.best_iteration
print(f"\nBest boosting round : {best_round}")
print("Training complete.\n")


target_pred       = xgb_model.predict(feature_test)
target_pred_proba = xgb_model.predict_proba(feature_test)[:, 1]


print("classification report")
print(classification_report(target_test, target_pred,
                             target_names=["No Stroke", "Stroke"]))

roc_auc = roc_auc_score(target_test, target_pred_proba)
print(f"ROC-AUC Score : {roc_auc:.4f}\n")

# With heavy class imbalance, the default 0.5 threshold often
# misses true stroke cases (low recall).  Try a lower threshold
# to trade some precision for higher recall on the positive class.

THRESHOLD = 0.3
target_pred_adjusted = (target_pred_proba >= THRESHOLD).astype(int)

print(f"classification report (threshold = {THRESHOLD})")
print(classification_report(target_test, target_pred_adjusted,
                             target_names=["No Stroke", "Stroke"]))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("XGBoost Stroke Prediction", fontsize=15, fontweight="bold")

#Confusion Matrix
cm = confusion_matrix(target_test, target_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Stroke", "Stroke"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix (threshold={THRESHOLD})")

#ROC curve
fpr, tpr, thresholds = roc_curve(target_test, target_pred_proba)
axes[1].plot(fpr, tpr, color="darkorange", lw=2,
             label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
# Mark the chosen threshold on the curve
threshold_idx = np.argmin(np.abs(thresholds - THRESHOLD))
axes[1].scatter(fpr[threshold_idx], tpr[threshold_idx],
                color="red", zorder=5, label=f"Threshold = {THRESHOLD}")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")

#Top-15 Feature Importances
importances = pd.Series(xgb_model.feature_importances_,
                        index=feature_matrix.columns)
top15 = importances.nlargest(15).sort_values()
top15.plot(kind="barh", ax=axes[2], color="darkorange")
axes[2].set_title("Top 15 Feature Importances")
axes[2].set_xlabel("Importance score")

plt.tight_layout()
plt.savefig("xgboost_results.png", dpi=150, bbox_inches="tight")
plt.show()



evals_result = xgb_model.evals_result()
val_auc = evals_result["validation_0"]["auc"]

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(val_auc) + 1), val_auc,
         color="darkorange", lw=1.5, label="Validation AUC")
plt.axvline(best_round, color="red", linestyle="--",
            label=f"Best round ({best_round})")
plt.xlabel("Boosting round")
plt.ylabel("AUC")
plt.title("XGBoost Validation AUC per Round")
plt.legend()
plt.tight_layout()
plt.savefig("xgboost_auc_curve.png", dpi=150, bbox_inches="tight")
plt.show()
