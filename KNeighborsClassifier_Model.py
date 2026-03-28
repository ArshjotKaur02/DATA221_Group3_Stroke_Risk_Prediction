# Model: K-Nearest Neighbors Classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

#Importing processed data from Stroke_Data_Processing.py
from Stroke_Data_Processing import feature_train, feature_test, target_train, target_test

# --------------------------------------------- FEATURE SCALING --------------------------------------------------------

# Some input features have very different ranges, for instance age vs glucose level. As a result one feature with bigger
# range will dominate computations, specifically for distance based models. Feature scaling will put features on a
# comparable scale around 0.

scaler = StandardScaler()
feature_train_scaled = scaler.fit_transform(feature_train)  #Learn mean/std from train set and then scale it
feature_test_scaled  = scaler.transform(feature_test)

# --------------------------------------------- TRAIN THE KNN MODEL ----------------------------------------------------

# Choosing k = 3 for a balance of accuracy, f1-score and recall
# Using metric = Euclidean to measure straight-line distance
kNeighbour_model = KNeighborsClassifier(n_neighbors = 3, metric = "euclidean")
kNeighbour_model.fit(feature_train_scaled, target_train)
predicted_labels = kNeighbour_model.predict(feature_test_scaled)
predicted_probability = kNeighbour_model.predict_proba(feature_test_scaled)[:,1]


# --------------------------------------------- EVALUATION METRICS -----------------------------------------------------

#Since the dataset is very imbalanced, the model might not predicts any positive cases, so using zero_division=0 prevents a runtime
knn_accuracy_score = accuracy_score(target_test, predicted_labels)
knn_precision = precision_score(target_test, predicted_labels, zero_division=0)
knn_recall = recall_score(target_test, predicted_labels, zero_division=0)
knn_f1_score = f1_score(target_test, predicted_labels, zero_division=0)
knn_roc_auc_score   = roc_auc_score(target_test, predicted_probability)
knn_confusion_matrix = confusion_matrix(target_test, predicted_labels)

print("-" * 45) # Visual seperator
print("-*- KNN PERFORMANCE METRICS -*-")
print("-" * 45)

print(f"Accuracy  : {knn_accuracy_score:.4f}")
print(f"Precision : {knn_precision:.4f}")
print(f"Recall    : {knn_recall:.4f}")
print(f"F1-Score  : {knn_f1_score:.4f}")
print(f"ROC-AUC   : {knn_roc_auc_score:.4f}")

print("\nConfusion Matrix:")
print(knn_confusion_matrix)
