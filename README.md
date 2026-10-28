# Machine Learning Classification Projects

A beginner-friendly portfolio of classification workflows built with Python, pandas, and scikit-learn. The repository starts with synthetic practice data and then applies an end-to-end workflow to the real Titanic training dataset.

## Projects

### 1. Classification with Pipelines

**File:** `classification_with_pipeline.py`

Demonstrates preprocessing and model training with scikit-learn `Pipeline` and `ColumnTransformer`, including missing-value imputation, scaling, categorical encoding, cross-validation, hyperparameter tuning, and evaluation.

### 2. Classification without Pipelines

**File:** `classification_without_pipeline.py`

Demonstrates explicit preprocessing steps, feature engineering, model training, and evaluation without combining every step into a single pipeline.

**Important limitation:** These first two projects use synthetic practice data whose target labels are generated randomly rather than meaningfully from the features. Their purpose is to practise the workflow, not to demonstrate real predictive value. The two scripts also use different preprocessing choices, so their scores are not a controlled comparison.

### 3. Titanic Survival Classification

**Project guide:** [`titanic/README.md`](titanic/README.md)  
**Script:** [`titanic/titanic_classification_ml_project.py`](titanic/titanic_classification_ml_project.py)

This project uses the Kaggle Titanic training dataset to predict whether a passenger survived.

Workflow:
- Exploratory data analysis (EDA), including missing values, target distribution, survival rates by passenger characteristics, and numerical correlations.
- Separation of features and target, followed by a stratified train/test split.
- Numerical preprocessing with median imputation and standard scaling.
- Categorical preprocessing with most-frequent imputation and one-hot encoding.
- A `ColumnTransformer` and model pipelines.
- Comparison of Logistic Regression and Random Forest using 5-fold cross-validation with F1 as the scoring metric.
- Hyperparameter tuning of the better baseline candidate with `GridSearchCV`.
- Held-out test evaluation using accuracy, precision, recall, F1, ROC-AUC, a classification report, and a confusion matrix.
- Saving the fitted preprocessing-and-model pipeline with joblib.

The baseline excludes `PassengerId`, `Name`, `Ticket`, and `Cabin`; those columns could be explored for feature engineering in future iterations. The EDA explores family size, but that feature is not currently included in the baseline model.

## Repository structure

```text
machine-learning-classification-projects/
├── classification_with_pipeline.py
├── classification_without_pipeline.py
├── titanic/
│   ├── README.md
│   └── titanic_classification_ml_project.py
├── requirements.txt
├── README.md
└── .gitignore
```

The Titanic CSV and generated model file are intentionally not committed. Download the labelled training data from Kaggle and place it at `titanic/titanic_dataset.csv` before running the Titanic script.

## Setup

Python 3.10 or newer is recommended.

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run the projects

From the repository root, run either synthetic-data script:

```bash
python classification_with_pipeline.py
python classification_without_pipeline.py
```

For the Titanic project, first place the Kaggle **labelled training CSV** at `titanic/titanic_dataset.csv`, then run:

```bash
python titanic/titanic_classification_ml_project.py
```

The Titanic script saves `titanic_classification_ml_project.pkl` in the current working directory. Run the command above from the repository root so generated files are easy to locate.

## Skills practised

- Exploratory data analysis
- Missing-value handling
- Categorical encoding and numerical scaling
- Train/test splitting and stratification
- scikit-learn pipelines and `ColumnTransformer`
- Cross-validation and hyperparameter tuning
- Classification metrics
- Feature selection and early-stage feature-engineering exploration
- Saving a fitted model pipeline with joblib

## Notes

- Evaluation results can vary with data, library versions, and modelling choices.
- Keep the held-out test set for final evaluation rather than repeatedly using it to choose features or models.
- This is a learning portfolio project, not a production-ready or deployed prediction service.

## Author

**Owais Ali**

Aspiring Machine Learning / AI Engineer

GitHub: https://github.com/oz-owaiz
