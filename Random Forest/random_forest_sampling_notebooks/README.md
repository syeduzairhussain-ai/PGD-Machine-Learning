# Random Forest Sampling Notebooks

This package contains two Jupyter notebooks:

1. `01_random_forest_feature_sampling.ipynb`
   - Demonstrates random feature/column sampling.
   - Shows how `max_features` works in Random Forest.

2. `02_random_forest_row_sampling.ipynb`
   - Demonstrates random row sampling.
   - Shows bootstrap sampling using `replace=True`.
   - Shows how `bootstrap=True` works in Random Forest.

Dataset:
- `dental_random_forest_data.csv`

Install requirements:

```bash
pip install pandas scikit-learn notebook
```

Run:

```bash
jupyter notebook
```

Important:
This is an educational dental dataset only. It is not for real medical or dental diagnosis.
