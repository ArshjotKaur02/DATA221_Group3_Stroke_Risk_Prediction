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