"""
Dataset: Stroke Prediction Dataset from Kaggle
Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

The data contains 5,110 patient records with 11 input features and one binary target variable (stroke).

Features include:
 - Numeric: age, avg_glucose_level, bmi
 - Binary: hypertension, heart_disease, ever_married
 - Categorical: gender, work_type, Residence_type, smoking_status

The dataset is highly imbalanced, with roughly 4.9% (249)positive stroke cases.
"""

# Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Using pd.read_csv() to reads stroke data CSV file into a DataFrame
patient_records_dataframe = pd.read_csv("healthcare_stroke_data.csv")
print(patient_records_dataframe.head())

# The id column in the dataset is just an identifier. It does not have any medical significance.
# If we include it, the model might try to "learn" from it, which makes no sense.
patient_records_dataframe = patient_records_dataframe.drop(columns=["id"])

# ---------------------------------------- HANDLING MISSING DATA -------------------------------------------------------

# bmi column has some N/A strings stored as text. Around 201 inputs are missing
# We'll convert those N/A strings to actual NaN (Not a Number) and then replace the as missing values with median.
#Information on .to_numeric() was accessed from: https://www.geeksforgeeks.org/python/python-pandas-to_numeric-method/

patient_records_dataframe["bmi"] = pd.to_numeric(patient_records_dataframe["bmi"], errors="coerce")
#errors="coerce" ensures that if parsing is invalid (for N/A in this case) then set that cell as NaN.

# We'll use median to fill the missing data because it is resistant to outliers. This makes it a better measure of
# the typical BMI value in the dataset.

# We avoid deleting rows with missing BMI values as it will result in loss of significant chunk of data and using mean
# is not ideal either as it is very sensitive to outliers and can make the imputed values less representative of the
# overall data.

median_bmi_value = patient_records_dataframe["bmi"].median()
# Using fillna() to replaces all NaN values with the median BMI value
patient_records_dataframe["bmi"] = patient_records_dataframe["bmi"].fillna(median_bmi_value)


