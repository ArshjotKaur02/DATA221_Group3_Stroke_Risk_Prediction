import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from Stroke_Data_Processing import feature_train, feature_test, target_train, target_test

"""
The model will be evaluated using the same metrics as all other models for fair comparison.

zero_division=0 prevents runtime warnings in rare cases where a model predicts no positive class at all since this is an imbalanced datasets
"""
def model_evaluate(model_name, trained_model, test_features, true_labels):
    predicted_labels = trained_model.predict(test_features)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print('-' * 50)
    print(f'-*- {model_name} PERFORMANCE METRICS -*-')
    print('-' * 50)
    print(f'Accuracy score  : {accuracy:.4f}')
    print(f'Precision score : {precision:.4f}')
    print(f'Recall score    : {recall:.4f}')
    print(f'F1-Score        : {f1:.4f}')
    print('\nConfusion Matrix:')
    print(conf_matrix)
    print()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

# Model setup: class_weight="balanced" increases penalty for minority class errors, help the model pay attention to rare stroke cases
decision_tree_model = DecisionTreeClassifier(random_state=4, class_weight="balanced")
decision_tree_model.fit(feature_train, target_train) # train the model on full training split
decision_tree_model_results = model_evaluate("Decision Tree", decision_tree_model, feature_test, target_test)

# Hyperparameter Tuning
# Step 1: Split training set into a smaller subset for training and validation
# Step 2: Train on multiple parameters combinations
# Step 3: Select the best setting based on validation recall score
# Step 4: Refit the chosen model on the original full training set
# Learn about this on https://www.geeksforgeeks.org/machine-learning/how-to-tune-a-decision-tree-in-hyperparameter-tuning/

feature_train_sub, feature_validation, target_train_sub, target_validation = train_test_split(
    feature_train, target_train, test_size=0.3, random_state=42, stratify=target_train)

parameter_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_split": [2, 10, 20, 40],  # Larger values to reduce overfitting
    "min_samples_leaf": [1, 5, 10, 20],  # Larger leaves to create smoother rules
}

# Track best model settings from validation performance
best_validation_recall = -1.0
best_validation_f1 = -1.0 # Used only when recall ties
best_parameters = None # Stores best parameter found

for criterion in parameter_grid["criterion"]:
    for max_depth in parameter_grid["max_depth"]:
        for min_samples_split in parameter_grid["min_samples_split"]:
            for min_samples_leaf in parameter_grid["min_samples_leaf"]:
                trial_tree_model = DecisionTreeClassifier(
                    criterion=criterion, max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced", random_state=42)

                trial_tree_model.fit(feature_train_sub, target_train_sub)
                validation_prediction = trial_tree_model.predict(feature_validation)
