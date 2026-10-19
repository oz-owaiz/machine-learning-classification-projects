# Machine Learning Classification Projects

Two beginner-friendly Python projects that explore a classification workflow using **scikit-learn**: one uses pipelines, while the other performs preprocessing manually.

## Projects

### 1. Classification with Pipelines
File: `classification_with_pipeline.py`

This script demonstrates how to organize preprocessing and model training with scikit-learn pipelines.

- Creates a synthetic dataset with 1,000 rows.
- Introduces missing values to practise data cleaning.
- Splits data into training and test sets.
- Imputes missing numeric values with the median and categorical values with the most frequent value.
- Scales numeric features and one-hot encodes categorical features.
- Combines preprocessing with Logistic Regression and Random Forest classifiers.
- Uses cross-validation to compare candidate models and `GridSearchCV` to tune hyperparameters.
- Evaluates the selected model using accuracy, precision, recall, F1 score, ROC-AUC, a classification report, and a confusion matrix.
- Saves the trained pipeline to `classification_pipeline.pkl` using joblib.

### 2. Classification without Pipelines
File: `classification_without_pipeline.py`

This script shows the individual preprocessing steps so they can be studied separately.

- Creates a synthetic dataset and introduces missing values.
- Splits data into training and test sets.
- Handles missing numeric and categorical values with `SimpleImputer`.
- Encodes education levels and one-hot encodes city categories.
- Creates example engineered features.
- Scales the data with `StandardScaler`.
- Trains and evaluates Logistic Regression and Random Forest classifiers.
- Reports accuracy, precision, recall, F1 score, ROC-AUC, classification reports, and confusion matrices.
- Saves the processed arrays as NumPy `.npy` files.

## Key Learning Goals

- Understand the difference between manual preprocessing and a pipeline-based workflow.
- Practise handling missing data and encoding categorical features.
- Learn why transformations should be fitted on training data and then applied to test data.
- Explore model comparison, cross-validation, hyperparameter tuning, and classification metrics.
- See how a trained pipeline or processed data can be saved for later use.

## Requirements

- Python 3.10 or newer recommended
- NumPy
- pandas
- scikit-learn
- joblib

## How to Run

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run either script:

   ```bash
   python classification_with_pipeline.py
   ```

   or

   ```bash
   python classification_without_pipeline.py
   ```

The scripts print preprocessing information and evaluation results. They also create output files in the folder from which they are run.

## Important Note About the Dataset

These projects use **synthetic practice data**, not a real-world dataset. The target label is generated randomly, so the features do not intentionally explain the target. As a result, model scores are for practising the workflow only and should not be interpreted as evidence of a useful real-world predictive model.

## Next Steps

- Fix and validate the workflow, then compare the results from both approaches.
- Apply similar techniques to a real dataset and document its limitations.
- Add reproducible results and lessons learned as the projects develop.
