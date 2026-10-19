# Machine Learning Classification Projects

A beginner-friendly portfolio project exploring two ways to build a machine learning classification workflow with Python and scikit-learn: **using pipelines** and **performing preprocessing manually**.

## Project Overview

The goal is to understand how data preprocessing, model training, and evaluation fit together—and why scikit-learn pipelines can make a workflow easier to manage.

### Project 1: Classification with Pipelines

**File:** `classification_with_pipeline.py`

This script builds a workflow using `Pipeline` and `ColumnTransformer`.

It demonstrates:
- Creating a synthetic dataset with 1,000 samples and introducing missing values.
- Splitting data into training and test sets.
- Imputing missing numerical values with the median and categorical values with the most frequent value.
- Scaling numerical features with `StandardScaler`.
- Encoding categorical features with `OneHotEncoder`.
- Combining preprocessing and classification in a single pipeline.
- Comparing Logistic Regression and Random Forest with 5-fold cross-validation.
- Tuning the selected model with `GridSearchCV`.
- Evaluating the final model with accuracy, precision, recall, F1 score, ROC-AUC, a classification report, and a confusion matrix.
- Saving the fitted pipeline to `classification_pipeline.pkl` with joblib.

### Project 2: Classification without Pipelines

**File:** `classification_without_pipeline.py`

This script performs preprocessing in separate, explicit steps to make each transformation easier to inspect.

It demonstrates:
- Creating a synthetic dataset and introducing missing values.
- Splitting data into training and test sets.
- Imputing missing values in numerical and categorical columns.
- Mapping education levels to ordered numeric values and one-hot encoding city categories.
- Creating example engineered features.
- Replacing invalid values and imputing any resulting missing values.
- Scaling features with `StandardScaler`.
- Training Logistic Regression and Random Forest classifiers.
- Evaluating both models with accuracy, precision, recall, F1 score, ROC-AUC, classification reports, and confusion matrices.
- Saving the processed training and test arrays as NumPy `.npy` files.

## Skills Practised

- Data cleaning and missing-value imputation
- Categorical encoding and numerical scaling
- Train/test splitting
- Feature engineering
- scikit-learn `Pipeline` and `ColumnTransformer`
- Cross-validation and hyperparameter tuning
- Classification evaluation metrics
- Saving trained models and processed data

## Repository Structure

```text
machine-learning-classification-projects/
├── classification_with_pipeline.py
├── classification_without_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

Generated model and data files are created when the scripts run; they are not required to be present in the repository beforehand.

## Requirements

- Python (3.10 or newer recommended)
- NumPy
- pandas
- scikit-learn
- joblib

## Run the Projects

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the required libraries:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run either project:

   ```bash
   python classification_with_pipeline.py
   ```

   ```bash
   python classification_without_pipeline.py
   ```

The first script saves a fitted pipeline as `classification_pipeline.pkl`. The second script saves processed feature and target arrays as `.npy` files.

## Dataset and Limitations

Both scripts use **synthetic practice data**, not a real-world dataset. The target labels are generated randomly and are not designed to depend on the input features. Therefore, the metrics are useful for practising the mechanics of a classification workflow, but they do **not** demonstrate real predictive value or business impact.

The two scripts also use slightly different preprocessing choices and feature-engineering steps, so their evaluation scores should not be treated as a controlled, apples-to-apples comparison of pipelines versus manual preprocessing.

## Future Improvements

- Apply the workflow to a real dataset with a meaningful prediction target.
- Compare both approaches on the same data split and evaluation setup.
- Document results, limitations, and lessons learned from a real-world use case.
