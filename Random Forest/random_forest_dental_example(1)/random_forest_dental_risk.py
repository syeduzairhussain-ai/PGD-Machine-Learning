"""
Random Forest Dental Risk Prediction Example
--------------------------------------------
This program predicts a patient's dental risk level using Random Forest.

Dataset file:
dental_risk_data.csv

Features:
1. age
2. brushing_per_day
3. sugar_intake_level
4. gum_bleeding
5. tooth_pain
6. last_dental_visit_months

Target:
dental_risk

Important:
This is a teaching example only. It is not for real medical diagnosis.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1: Load the dataset
data = pd.read_csv("dental_risk_data.csv")

print("Dataset Preview:")
print(data.head())


# Step 2: Separate input features and output target
X = data[[
    "age",
    "brushing_per_day",
    "sugar_intake_level",
    "gum_bleeding",
    "tooth_pain",
    "last_dental_visit_months"
]]

y = data["dental_risk"]


# Step 3: Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Step 4: Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Step 5: Train the model
model.fit(X_train, y_train)


# Step 6: Make predictions on test data
y_pred = model.predict(X_test)


# Step 7: Check model performance
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 8: Predict dental risk for a new patient
new_patient = pd.DataFrame({
    "age": [35],
    "brushing_per_day": [1],
    "sugar_intake_level": [4],
    "gum_bleeding": [1],
    "tooth_pain": [1],
    "last_dental_visit_months": [24]
})

prediction = model.predict(new_patient)

print("\nNew Patient Data:")
print(new_patient)

print("\nPredicted Dental Risk:", prediction[0])


# Step 9: Show feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance)
