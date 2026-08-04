"""
Random Forest Dental Treatment Prediction with Visualization
------------------------------------------------------------
This is a beginner-friendly Random Forest classification example.

The model predicts dental treatment needed:
1. No Treatment
2. Filling
3. Root Canal

Important:
This is an educational machine learning example only.
It is not for real dental diagnosis.
"""

# ===============================
# 1. Import required libraries
# ===============================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import plot_tree


# ===============================
# 2. Load the dataset
# ===============================

data = pd.read_csv("dental_treatment_data.csv")

print("Dataset Preview:")
print(data.head())


# ===============================
# 3. Visualize class distribution
# ===============================

plt.figure(figsize=(7, 5))
data["treatment_needed"].value_counts().plot(kind="bar")
plt.title("Dental Treatment Class Distribution")
plt.xlabel("Treatment Needed")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()


# ===============================
# 4. Select features and target
# ===============================

X = data[
    [
        "age",
        "tooth_pain_level",
        "gum_bleeding",
        "cavity_present",
        "plaque_level",
        "last_dental_visit_months"
    ]
]

y = data["treatment_needed"]


# ===============================
# 5. Split data into train and test
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ===============================
# 6. Create Random Forest model
# ===============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ===============================
# 7. Train the model
# ===============================

model.fit(X_train, y_train)


# ===============================
# 8. Make predictions
# ===============================

y_pred = model.predict(X_test)


# ===============================
# 9. Evaluate model performance
# ===============================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# 10. Visualize confusion matrix
# ===============================

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix - Dental Treatment Prediction")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# ===============================
# 11. Visualize feature importance
# ===============================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(8, 5))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# ===============================
# 12. Visualize one decision tree
# ===============================

plt.figure(figsize=(20, 10))

plot_tree(
    model.estimators_[0],
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("One Decision Tree from Random Forest")
plt.tight_layout()
plt.savefig("one_decision_tree.png", dpi=200)
plt.show()


# ===============================
# 13. Predict treatment for a new patient
# ===============================

new_patient = pd.DataFrame({
    "age": [32],
    "tooth_pain_level": [7],
    "gum_bleeding": [1],
    "cavity_present": [1],
    "plaque_level": [7],
    "last_dental_visit_months": [24]
})

prediction = model.predict(new_patient)

print("\nNew Patient Data:")
print(new_patient)

print("\nPredicted Dental Treatment:", prediction[0])
