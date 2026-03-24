# Model: Random Forest Classifier

"""
#For the team to review:

Random Forest is an ENSEMBLE method - it builds many Decision Trees and combines their predictions.

Two sources of randomness:

1) BOOTSTRAP SAMPLING (Bagging):
   Each tree is trained on a RANDOM SUBSET of training data. "With replacement" means the same row can be picked
   multiple times. This creates slightly different training sets for each tree. Trees are diverse so they don't
   all make the same mistakes.

2) RANDOM FEATURE SELECTION:
   At each split point, a tree only considers a RANDOM SUBSET of features. This further decorrelates the trees,
   making the forest more robust.

Final Prediction:
   Each tree votes for class 0 or 1. The class that gets the most votes wins.
   Example: 100 trees, 63 vote "stroke", the prediction is stroke.

Why Random Forest is better than Decision tree:
   A single Decision Tree overfits (memorizes training data, bad on new data). Random Forest reduces overfitting
   by averaging across many trees. Each tree is wrong in a different way and errors cancel out.
   Result: much better generalization to new data.

Information accessed from: https://www.youtube.com/watch?v=Wj3qfSyRHys
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Importing processed data from Stroke_Data_Processing.py
from Stroke_Data_Processing import feature_train, feature_test, target_train, target_test

# ----------------------------------------- TRAIN THE RANDOM FOREST MODEL ----------------------------------------------

# n_estimators = 100
#   Counts of decision trees. More trees means more stable predictions, but slower to train

# max_depth = 10
#   Limiting depth acts as regularization to prevents overfitting

# max_features= "sqrt"
#   Number of features randomly selected at each split.
#   "sqrt" means square root of total features. For 10 features → ~3 features per split
#   This is one of the two randomness that makes trees different from each other

# class_weight = "balanced"
#   "balanced" adjusts the weight of each class inversely to its frequency
#   This prevents the model from ignoring the minority class, as in this case class 1

# random_state=42
#   Setting this ensures reproducible results every run.

# Information on training Random Forest was accessed from:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

random_forest_model = RandomForestClassifier(n_estimators = 100, max_depth= 10, max_features="sqrt",
                                             class_weight="balanced", random_state=42)

random_forest_model.fit(feature_train, target_train)
predicted_labels = random_forest_model.predict(feature_test)

# --------------------------------------------- EVALUATION METRICS -----------------------------------------------------

random_forest_accuracy = accuracy_score(target_test, predicted_labels)
random_forest_precision = precision_score(target_test, predicted_labels)
random_forest_recall = recall_score(target_test, predicted_labels)
random_forest_f1_score = f1_score(target_test, predicted_labels)
random_forest_confusion_matrix = confusion_matrix(target_test, predicted_labels)

print("-" * 45) # Visual seperator
print("-*- RANDOM FOREST PERFORMANCE METRICS -*-")
print("-" * 45)

print(f"Accuracy  : {random_forest_accuracy:.4f}")
print(f"Precision : {random_forest_precision:.4f}")
print(f"Recall    : {random_forest_recall:.4f}")
print(f"F1-Score  : {random_forest_f1_score:.4f}")

print("\nConfusion Matrix:")
print(random_forest_confusion_matrix)