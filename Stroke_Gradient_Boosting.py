"""
Stroke Prediction - XGBoost Model
Imports preprocessed splits directly from Stroke_Data_Processing.py
"""

import xgboost
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from Stroke_Data_Processing import (
    feature_train, feature_test,
    target_train, target_test
)

negative_count = (target_train == 0).sum()
positive_count = (target_train == 1).sum()
scale_positive_weight = negative_count / positive_count

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

#Paramaters adjusted for higher recall while maintaining other metrics
xgboost_model = xgboost.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.8,
    min_child_weight=20,
    gamma=0,
    reg_alpha=0.5,
    reg_lambda=2.0,
    scale_pos_weight=scale_positive_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42
)

xgboost_model.fit(
    feature_train, target_train,
    eval_set=[(feature_test, target_test)],
    verbose=False
)

target_prediction = xgboost_model.predict(feature_test)
target_prediction_probability = xgboost_model.predict_proba(feature_test)[:, 1]


# A lower threshold improves recall on the minority stroke class
# at the cost of more false positives
THRESHOLD = 0.3
target_prediction_adjusted = (target_prediction_probability >= THRESHOLD).astype(int)

xgboost_accuracy = accuracy_score(target_test, target_prediction_adjusted)
xgboost_precision = precision_score(target_test, target_prediction_adjusted)
xgboost_recall = recall_score(target_test, target_prediction_adjusted)
xgboost_f1_score = f1_score(target_test, target_prediction_adjusted)
xgboost_confusion_matrix = confusion_matrix(target_test, target_prediction_adjusted)
xgboost_roc_auc = roc_auc_score(target_test, target_prediction_probability)



print(f"Accuracy  : {xgboost_accuracy:.4f}")
print(f"Precision : {xgboost_precision:.4f}")
print(f"Recall    : {xgboost_recall:.4f}")
print(f"F1-Score  : {xgboost_f1_score:.4f}")
print(f"Confusion Matrix:\n{xgboost_confusion_matrix}")
print(f"ROC-AUC   : {xgboost_roc_auc:.4f}")
