# Stroke Risk Prediction - DATA 221 Group Project

Predicting whether a patient is at risk of stroke using six machine learning models trained on the 
[Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

## The Problem

Stroke is one of the leading causes of death and long-term disability worldwide. Early identification of high-risk 
patients allows for preventive intervention. This project builds and compares different ML classifiers on a dataset of 
5,110 patient records to determine which model best identifies stroke risk from routine health indicators.

The core challenge is class imbalance. Only around 4.9% of patients (249 of 5,110) experienced a stroke. A naive model 
that always predicts "no stroke" would achieve approximately 95% accuracy, while being completely useless medically. 
We tried to address this issue in data processing.

## Dataset

| Property | Value                                |
|---|--------------------------------------|
| Source | Kaggle: stroke-prediction-dataset    |
| Records | 5,110 patients                       |
| Features | 11 input features + 1 target         |
| Target | `stroke` (0 = no stroke, 1 = stroke) |
| Imbalance | ~4.9% positive (stroke) cases        |

**Features used:**
- Numeric: `age`, `avg_glucose_level`, `bmi`
- Binary: `hypertension`, `heart_disease`, `ever_married`, `Residence_type`
- Categorical: `gender`, `work_type`, `smoking_status`


## Project Structure
```
- Stroke_Data_Processing.py      # Includes data cleanup and train/test split
- LogisticRegression_Model.py    # Model 1: Logistic Regression
- KNeighborsClassifier_Model.py  # Model 2: K-Nearest Neighbors
- DecisionTree_Model.py          # Model 3: Decision Tree
- Neural_Net_Model.py            # Model 4: Neural Network
- RandomForest_Model.py          # Model 5: Random Forest
- Stroke_Gradient_Boosting.py    # Model 6: Gradient Boosting
- healthcare_stroke_data.csv     # Raw dataset (place in same folder)
- README.md                      # This file
```

## Data Processing

| Step                   | What                                    | Why                                                                                                         |
|------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Drop `id` column       | Remove row identifier                   | It has no medical meaning; keeping it lets the model "memorise" row numbers                                 |
| Missing data values    | Replace text "N/A" with median          | There were around 201 missing values. Median is used since it is resistant to outliers unlike mean          |
| Binary encoding        | `ever_married`, `Residence_type` to 0/1 | Two-category columns need no ordering; direct mapping to 0 and 1                                            |
| One-Hot Encoding       | `gender`, `work_type`, `smoking_status` | More than 2 categories. Assigning numbers (1,2,3) would imply a false order                                 |
| 70/30 stratified split | `train_test_split` with `stratify`      | Ensures the 4.9% stroke proportion is preserved in both splits; `random_state=42` makes every run identical |


## Evaluation Metrics

### Why accuracy is misleading here

If a model predicts "no stroke" for every patient, it gets **95% accuracy** while catching zero stroke cases. Accuracy 
is useless as a primary metric on imbalanced data.

### The metrics that matter

| Metric | Formula | What it tells you                                                                                     |
|---|---|-------------------------------------------------------------------------------------------------------|
| **Recall** | TP / (TP + FN) | Of all actual stroke patients, how many did we catch? Most important as a missed stroke can be fatal. |
| **Precision** | TP / (TP + FP) | Of all patients we flagged, how many truly had a stroke?                                              |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall. Useful single summary.                                         |
| **ROC-AUC** | Area under ROC curve | Tests the model at every possible threshold. It is not affected by class imbalance.                   |

### Confusion Matrix 

**Confusion Matrix** is a snapshot at one specific threshold.

```
                  Predicted NO    Predicted YES
Actual NO  :         TN               FP   (false alarm)
Actual YES :         FN               TP   (caught stroke)
                  (missed stroke)
```

## Limitations

- **Dataset size**: 5,110 records is small for medical ML. Models may not generalise well to different populations.
- **Class imbalance**: Even with weighting, 249 positive cases limits how much the model can learn about stroke patterns.
- **Missing data**: 201 BMI values were imputed with the median. Real patient BMI values may differ.
- **No external validation**: All evaluation is on one held-out test set from the same dataset. Performance on new patients is unknown.


## Partner Contributions

| Member | Contribution                      |
|---|-----------------------------------|
| A. Kaur | KNN, Random Forest                |
| A. Spring | Logistic Regression               |
| L. Nguyen | Decision Tree                     |
| U. Haris | Neural Network, Gradient Boosting |

## References

- Stroke Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- GeeksforGeeks - `pd.to_numeric()`: https://www.geeksforgeeks.org/python/python-pandas-to_numeric-method/
- GeeksforGeeks - `pd.get_dummies()`: https://www.geeksforgeeks.org/pandas/python-pandas-get_dummies-method/
- Random Forest explanation: https://www.youtube.com/watch?v=Wj3qfSyRHys
- Hyperparameter tuning Decision Trees: https://www.geeksforgeeks.org/machine-learning/how-to-tune-a-decision-tree-in-hyperparameter-tuning/