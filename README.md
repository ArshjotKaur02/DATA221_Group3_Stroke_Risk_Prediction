# DATA221_Group3_Stroke_Risk_Prediction
Machine learning project for predicting stroke risk using health indicators, including data preprocessing, model comparison, and evaluation.


------------------------- LOGISTIC REGRESSION MODEL -------------------------
Background and Dataset:
The dataset used in this project is the Stroke Prediction Dataset sourced from 
Kaggle, containing 5,110 patient records. Each record has 11 input features and
one target variable indicating whether the patient had a stroke. The features 
cover a mix of numeric data including age, BMI, and average glucose level, binary
data including hypertension, heart disease, marital status, categorical data 
including gender, work type, residence type, and smoking status. The dataset is 
heavily imbalanced, with only 249 out of 5,110 patients having experienced a 
stroke, making up approximately 4.9% of the data.

Data Preprocessing:
Before training the model, several preprocessing steps were carried out to
ensure the data was clean and in a format the model could work with. The first
issue addressed was missing data. The BMI column contained approximately 201
missing values that were stored as the test "N/A" rather than actual empty cells.
These were first converted to proper missing values using pandas, and then filled
in using the median BMI of the dataset. The median was chosen over the mean because
it is not affected by extreme outlier values, making it a more reliable representation
of the typical BMI in the dataset. Rows with missing values were not deleted because
doing so would have removed a significant portion of the data. The second step 
involved encoding categorical variables. Columns with only two possible values, 
specifically marital status and residence type, were converted to binary numbers
where 1 represented Yes or Urban and 0 represented No or Rural. For columns with more
than two categories, specifically gender, work type, and smoking status, One-Hot Encoding
was applied. This technique creates a separate binary column for each category.
The first category in each group was dropped to avoid a redundancy issue known as the 
dummy variable trap, where one column can be perfectly predicted from the others.
Finally, the dataset was split into a training set containing 70% of the data and 
a test set containing the remaining 30%. The split was stratified, meaning the proportion
of stroke cases was kept the same in both sets, which is important given how imbalanced 
the data is. 

Model Design and Justification:
Logistic Regression was chosen as the modelling technique because it is well suited
to binary classification problems, in this case predicting either stroke or no
stroke. It is also interpretable, meaning we can directly examine the model's coefficients
to understand which features are driving predictions. Several important design 
decisions were made when setting up the model. Feature scaling was applied using a 
Standard Scaler, which transforms each feature to have a mean of zero and a standard
deviation of one. This step was necessary because Logistic Regression is sensitive
to features being on very different scales. For example, age ranges from 0 to 100 while
average glucose level ranges from roughly 50 to 280. Without scaling, the model's 
optimization process could be distorted. Critically, the scaler was fitted only on the 
training data and then applied to the test data, to prevent data leakage where
information from the test set influences the model. To address the class imbalance, 
the model was configured with balanced class weighting. This automatically adjusts the 
penalty the model receives for misclassifying each class, giving mush higher weight to the 
minority stroke class. Without this, the model would likely learn to predict no stroke
for every patient and still appear to be around 95% accurate, which would be completely 
useless in a real medical setting. The model was trained using the LBFGS solver, a 
quasi-Newton optimization algorithm that is efficient and reliable for datasets of 
this size. The maximum number of iterations was set to 1,000 to give the solver enough
steps to fully converge on a solution.

Results and Evaluation:
The model was evaluated using several metrics rather than just accuracy, because accuracy
alone is misleading on imbalanced datasets. The ROC-AUC score was 0.839, which measures 
the model's ability to correctly rank stroke patients above non-stroke patients across all
possible decision thresholds. A score of 0.5 would represent random guessing and a score of
1.0 would be perfect, so 0.839 indicates the model has strong discriminative ability. Looking 
at the classification report, the model achieved a recall of 79% on the stroke class, meaning 
it correctly identified 79 out of every 100 actual stroke cases. This is the most important
metric in a medical screening context because missing a real stroke case, known as a false
negative, carries far greater consequences than incorrectly flagging a healthy patient, known
as a false positive. The precision on the stroke class was 13%, which reflects the trade-off
made by using balanced class weighting. The model is intentionally cautious, raising
more alerts on order to avoid missing real cases. 

Feature Importance:
After scaling, the model's coefficients reveal which features had the greatest influence
on stroke predictions. Age had by far the highest coefficient at 1.90, meaning it was the 
single strongest predictor of stroke risk in the model. Average glucose level was the second
most impactful numeric feature with a coefficient of 0.20, followed by hypertension
at 0.16. Smoking status also appeared among the top features, with currently smoking increasing
risk and never having smoked decreasing it. These findings are consistent with established
medical research on stroke risk factors, which adds confidence that the model has learned 
meaningful patterns from the data rather than noise. 