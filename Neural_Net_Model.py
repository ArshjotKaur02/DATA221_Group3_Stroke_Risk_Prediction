import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from Stroke_Data_Processing import (
    feature_train, feature_test,
    target_train, target_test
)

# We fit the scaler ONLY on training data to avoid data leakage
# the test set is transformed using the training set's mean and std.

scaler = StandardScaler()
feature_train_scaled = scaler.fit_transform(feature_train)
feature_test_scaled  = scaler.transform(feature_test)

# Keras accepts a class_weight dictionary in model.fit() to
# penalise misclassification of the minority class more heavily.
# The recommended value is: total_samples / (num_classes * class_count)

negative_count = (target_train == 0).sum()
positive_count = (target_train == 1).sum()
total     = len(target_train)

class_weight = {
    0: total / (2 * negative_count),   # weight for no stroke
    1: total / (2 * positive_count)    # weight for stroke
}

# Architecture:
#   - Input layer matches the number of features
#   - Two hidden layers with ReLU activation
#   - BatchNormalization stabilises and speeds up training
#   - Dropout randomly deactivates neurons during training to
#     help combat overfitting
#   - Output layer uses sigmoid activation for binary classification,
#     outputting a probability between 0 and 1

num_features = feature_train_scaled.shape[1]

neural_net_model = Sequential([
    Dense(64, activation="relu", input_shape=(num_features,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])


neural_net_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



# EarlyStopping monitors validation loss and stops training if
# it does not improve for 20 consecutive epochs, then restores
# the weights from the best epoch.

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)




neural_net_model.fit(
    feature_train_scaled, target_train,
    epochs=200,
    batch_size=32,
    validation_data=(feature_test_scaled, target_test),
    class_weight=class_weight,
    callbacks=[early_stopping],
    verbose=1
)




target_prediction_probability = neural_net_model.predict(feature_test_scaled).flatten()

# A lower threshold improves recall on the minority stroke class
# at the cost of more false positives
THRESHOLD = 0.3
target_prediction_adjusted = (target_prediction_probability >= THRESHOLD).astype(int)


nn_accuracy         = accuracy_score(target_test, target_prediction_adjusted)
nn_precision        = precision_score(target_test, target_prediction_adjusted)
nn_recall           = recall_score(target_test, target_prediction_adjusted)
nn_f1_score         = f1_score(target_test, target_prediction_adjusted)
nn_confusion_matrix = confusion_matrix(target_test, target_prediction_adjusted)
nn_roc_auc = roc_auc_score(target_test, target_prediction_probability)


print(f"Accuracy  : {nn_accuracy:.4f}")
print(f"Precision : {nn_precision:.4f}")
print(f"Recall    : {nn_recall:.4f}")
print(f"F1-Score  : {nn_f1_score:.4f}")
print(f"Confusion Matrix:\n{nn_confusion_matrix}")
print(f"ROC-AUC   : {nn_roc_auc:.4f}")
