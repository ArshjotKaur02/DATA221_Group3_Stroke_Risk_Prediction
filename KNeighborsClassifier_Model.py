# Model: K-Nearest Neighbors Classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

#Importing processed data from Stroke_Data_Processing.py
from Stroke_Data_Processing import feature_train, feature_test, target_train, target_test

# --------------------------------------------- FEATURE SCALING --------------------------------------------------------

# Some input features have very different ranges, for instance age vs glucose level. As a result one feature with bigger
# range will dominate computations, specifically for distance based models. Feature scaling will put features on a
# comparable scale around 0.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(feature_train)  #Learn mean/std from train set and then scale it

#Testing various k values for maximum performance
accuracies = []
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(feature_train, target_train)
    predicted = knn.predict(feature_test)

    accuracy = accuracy_score(target_test, predicted)
    accuracies.append(accuracy)

print(accuracies)

plt.plot(k_values, accuracies, '-o', label='Accuracy Score', color='orange')
plt.xlabel("K-Value")
plt.ylabel("Accuracy Score")
plt.legend()
plt.show()
