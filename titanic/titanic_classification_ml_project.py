# Titanic Survival Classification
# A beginner-friendly end-to-end classification workflow using pandas and scikit-learn.

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme()


# -------------------------------------------------------------------
# Step 1: Load the dataset
# -------------------------------------------------------------------
# Place the Kaggle training dataset in this folder as titanic_dataset.csv.
DATA_PATH = Path(__file__).resolve().parent / "titanic_dataset.csv"
df = pd.read_csv(DATA_PATH)


# -------------------------------------------------------------------
# Steps 2-11: Exploratory Data Analysis (EDA)
# -------------------------------------------------------------------
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe(include="all").T)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nColumns with missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nMissing-value percentages:")
print((df.isnull().mean() * 100).round(2))

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nSurvival counts (0 = did not survive, 1 = survived):")
print(df["Survived"].value_counts().sort_index())

print("\nSurvival percentages:")
print((df["Survived"].value_counts(normalize=True).sort_index() * 100).round(1))

sns.countplot(data=df, x="Survived", color="C0")
plt.title("Passenger Survival Counts")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of passengers")
plt.show()

print("\nSurvival rate by sex:")
print(df.groupby("Sex")["Survived"].mean().round(3))

sns.barplot(data=df, x="Sex", y="Survived", errorbar=None, color="C0")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival rate")
plt.show()

print("\nSurvival counts by sex:")
print(df.groupby("Sex")["Survived"].value_counts().sort_index())

sns.countplot(data=df, x="Sex", hue="Survived", palette="Set2")
plt.title("Survival Counts by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

print("\nAge statistics:")
print(df["Age"].describe())

sns.histplot(data=df, x="Age", bins=30, kde=True, color="C0")
plt.title("Distribution of Passenger Ages")
plt.show()

sns.boxplot(data=df, x="Survived", y="Age", color="C0")
plt.title("Age Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.show()

print("\nFare statistics:")
print(df["Fare"].describe())

sns.histplot(data=df, x="Fare", bins=50, kde=True, color="C0")
plt.title("Distribution of Ticket Fares")
plt.show()

sns.boxplot(data=df, x="Survived", y="Fare", color="C0")
plt.title("Ticket Fare by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.show()

eda_df = df.copy()
eda_df["FamilySize"] = eda_df["SibSp"] + eda_df["Parch"] + 1

print("\nSurvival rate by family size:")
print(eda_df.groupby("FamilySize")["Survived"].agg(["count", "mean"]).round(3))

sns.barplot(data=eda_df, x="FamilySize", y="Survived", errorbar=None, color="C0")
plt.title("Survival Rate by Family Size")
plt.ylabel("Survival rate")
plt.show()

print("\nSurvival rate by embarkation port:")
print(df.groupby("Embarked")["Survived"].mean().round(3))

sns.barplot(data=df, x="Embarked", y="Survived", errorbar=None, color="C0")
plt.title("Survival Rate by Embarkation Port")
plt.ylabel("Survival rate")
plt.show()

print("\nSurvival rate by passenger class:")
print(df.groupby("Pclass")["Survived"].mean().round(3))

sns.barplot(data=df, x="Pclass", y="Survived", errorbar=None, color="C0")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival rate")
plt.show()

numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numerical Correlation Heatmap")
plt.show()


# -------------------------------------------------------------------
# Step 12: Separate features (X) and target (y)
# -------------------------------------------------------------------
# PassengerId is an identifier. Name, Ticket, and Cabin are excluded from
# this baseline model; they could be explored for feature engineering later.
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
y = df["Survived"]


# -------------------------------------------------------------------
# Step 13: Train/test split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)


# -------------------------------------------------------------------
# Step 14: Define numerical and categorical columns
# -------------------------------------------------------------------
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]


# -------------------------------------------------------------------
# Steps 15-17: Preprocessing pipelines and ColumnTransformer
# -------------------------------------------------------------------
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# -------------------------------------------------------------------
# Step 18: Create model pipelines
# -------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    ),
}


# -------------------------------------------------------------------
# Step 19: Compare baseline models using 5-fold cross-validation
# -------------------------------------------------------------------
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1
    )
    cv_results[name] = scores.mean()
    print(f"\n{name}")
    print("F1 scores:", scores)
    print("Mean CV F1:", round(scores.mean(), 4))

best_model_name = max(cv_results, key=cv_results.get)
best_pipeline = models[best_model_name]
print("\nBest baseline candidate:", best_model_name)


# -------------------------------------------------------------------
# Step 20: Define hyperparameter search space
# -------------------------------------------------------------------
if best_model_name == "Logistic Regression":
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"],
    }
else:
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2],
    }


# -------------------------------------------------------------------
# Step 21: Tune the selected model with GridSearchCV
# -------------------------------------------------------------------
grid_search = GridSearchCV(
    estimator=best_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True,
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\nBest hyperparameters:")
print(grid_search.best_params_)
print("\nBest cross-validation F1:", round(grid_search.best_score_, 4))


# -------------------------------------------------------------------
# Step 22: Evaluate once on the held-out test set
# -------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nFinal test results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 score:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------------------------
# Step 23: Save the complete fitted pipeline
# -------------------------------------------------------------------
joblib.dump(best_model, "titanic_classification_ml_project.pkl")
print("\nTrained pipeline saved to titanic_classification_ml_project.pkl")
