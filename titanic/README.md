# Titanic Survival Classification

## Project Overview

This project uses the Kaggle Titanic training dataset to build a machine learning model that predicts whether a passenger survived. It is a learning project focused on practising a complete classification workflow with Python and scikit-learn.

## Workflow

1. **Exploratory Data Analysis (EDA):** Inspect dataset structure, data types, summary statistics, missing values, duplicates, survival distribution, and survival patterns by passenger characteristics.
2. **Prepare features and target:** Use `Survived` as the target. Exclude `PassengerId`, `Name`, `Ticket`, and `Cabin` from this baseline model.
3. **Train/test split:** Reserve a stratified 20% hold-out test set.
4. **Preprocessing:** Use median imputation and standard scaling for numeric features; use most-frequent imputation and one-hot encoding for categorical features.
5. **Build pipelines:** Combine preprocessing and model training with `Pipeline` and `ColumnTransformer`.
6. **Compare baseline models:** Compare Logistic Regression and Random Forest with 5-fold cross-validation, using F1 score.
7. **Tune hyperparameters:** Use `GridSearchCV` to tune the better baseline candidate.
8. **Evaluate:** Report accuracy, precision, recall, F1 score, ROC-AUC, classification report, and confusion matrix on the held-out test set.
9. **Save the model:** Save the fitted preprocessing-and-model pipeline with joblib.

## Features Used

- **Numeric:** `Age`, `SibSp`, `Parch`, `Fare`
- **Categorical:** `Pclass`, `Sex`, `Embarked`

The current baseline does not add engineered features. During EDA, the script explores family size (`SibSp + Parch + 1`), but this feature is not included in model training yet. Future iterations could test `FamilySize`, `IsAlone`, passenger titles extracted from `Name`, or useful information from `Cabin` and `Ticket`. These should be tested against the baseline rather than assumed to improve it.

## Dataset

Download the labelled training dataset from the [Kaggle Titanic: Machine Learning from Disaster competition](https://www.kaggle.com/competitions/titanic/data). You may need to sign in to Kaggle and accept the competition rules.

1. Download `train.csv` from Kaggle.
2. Place it in this folder.
3. Rename it to `titanic_dataset.csv`.

The expected path is:

```text
titanic/titanic_dataset.csv
```

The dataset is not included in this repository; download it from Kaggle yourself. Do not commit the CSV unless you have confirmed you are permitted to redistribute it.

## Requirements

Python 3.10 or newer is recommended. From the repository root, install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Project

From the repository root, after placing the CSV at the path shown above, run:

```bash
python titanic/titanic_classification_ml_project.py
```

The script prints cross-validation results, the selected model and best hyperparameters, and final test metrics. It also displays EDA charts and saves the fitted pipeline as `titanic_classification_ml_project.pkl` in the current working directory. Run the command from the repository root to keep the generated model file there.

## Important Notes

- The test set is held out from cross-validation and hyperparameter tuning and is used for final evaluation.
- The baseline excludes some original columns that may be useful for future feature-engineering experiments.
- Scores can vary with library versions and modelling choices; use the metrics produced by your own run rather than assuming a fixed result.
- This is an educational portfolio project, not a deployed production system.
