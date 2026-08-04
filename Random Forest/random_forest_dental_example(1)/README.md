# Random Forest Dental Example

This folder contains a simple Random Forest classification example for dental risk prediction.

## Files

1. `dental_risk_data.csv`  
   A small sample dataset for dental risk prediction.

2. `random_forest_dental_risk.py`  
   Python code to train and test a Random Forest model.

## Dataset Columns

- `age`
- `brushing_per_day`
- `sugar_intake_level`
  - 1 = low sugar intake
  - 5 = very high sugar intake
- `gum_bleeding`
  - 0 = No
  - 1 = Yes
- `tooth_pain`
  - 0 = No
  - 1 = Yes
- `last_dental_visit_months`
- `dental_risk`
  - Low
  - Medium
  - High

## How to Run

Install required libraries:

```bash
pip install pandas scikit-learn
```

Run the code:

```bash
python random_forest_dental_risk.py
```

Make sure the Python file and CSV file are in the same folder.

## Note

This is only a teaching example. It should not be used for real dental or medical diagnosis.
