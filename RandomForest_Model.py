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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Importing processed data from Stroke_Data_Processing.py
from Stroke_Data_Processing import feature_train, feature_test, target_train, target_test