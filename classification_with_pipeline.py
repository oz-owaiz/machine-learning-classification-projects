# ===================================================================
"""
In this lab session I will be working on the complete work flow
of Machine Learning - Classification. We will create pipelines to
Streamline the process of preprocessing after completing this lab session.
Our aim will be to move forward with real data so we can have experience
how it feels to work on a real world data.

It will not be just a code, but steps will be involved so everything looks
organized and professional.

The project is based on the the Machine Learning Classification Project.
It is based on the synthetic dataset containing information about people
such as their age, income, education, work experience, and city.


"""

# ===================================================================
# Step # 1: Import libraries
# ===================================================================
"""
I will be importing the which are relevant to project I will be working on.
"""
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ===================================================================
# Step # 2 : Creating the Dataset
# ===================================================================
"""
In the following step I will creating an artificial dataset through which
we will then practice our machine learning classification model.
"""
np.random.seed(42)
n_samples = 1000

data = {
    "age": np.random.randint(18, 80, n_samples),
    "income": np.random.normal(50000, 20000, n_samples),
    "education": np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"], n_samples
    ),
    "experience_years": np.random.randint(0, 40, n_samples),
    "city": np.random.choice(
        ["New York", "Los Angeles", "Chicago", "Houston"], n_samples
    ),
    "target": np.random.choice([0, 1], n_samples),
}

# In this step I have created the Dataframe using Pandas
df = pd.DataFrame(data)

print(df.head())
# Print the first 5 rows
print(df.tail())
# Print the last 5 rows
print(df.info())
# Print the information of Dataset
print(df.isnull().sum())
# Prints if there are any missing values in the Dataset


# ===================================================================
# Step # 3 : Introducing Missing Values
# ===================================================================
"""
Since the data we have does not have any missing values I will introduce
the missing values inside the dataset. So we can preprocess it train our model.

"""
df.loc[df.sample(frac=0.08, random_state=1).index, "income"] = np.nan
# Missing value introduced in the Income column

df.loc[df.sample(frac=0.05, random_state=2).index, "education"] = np.nan
# Missing value introduced in the Education column

df.loc[df.sample(frac=0.04, random_state=3).index, "city"] = np.nan
# Missing value introduced in the City column

df.loc[df.sample(frac=0.05, random_state=4).index, "experience_years"] = np.nan
# Missing value introduced in the Experience Years

# ===================================================================
# Step # 4 : Introducing Missing Values
# ===================================================================
"""
I will inspecting the dataset after introducing the missing values

"""
print("\n Dataset Shape:")
print(df.shape)

print("\nMissing Values per Column:")
print(df.isnull().sum())

print("\nPercentage of Missing Values:")
print(df.isnull().mean() * 100)

print("\nTarget Distribution:")
print(df["target"].value_counts())

# ===================================================================
# Step # 5 : Seperate Features (X) and Target (y)
# ===================================================================
"""
In this step I will be separating the dataset into features and targets
X will be our feature and y will be our target.
The reason we do this (X) holds the inputs we will be using and 
(y) holds the target which we will be predicting.
"""
X = df.drop(columns=["target"])
# We need to drop the column target because it is part of y which will be
# used for predicting
y = df["target"]

print("\nFeatures:")
print(X.head())  # First 5 features of the dataset
print(X.tail())  # Last 5 features of the dataset


print("\nTarget:")
print(y.head())  # First 5 Targets
print(y.tail())  # Last 5 Targets

# ===================================================================
# Step # 6 : Splitting into Training and Testing
# ===================================================================
"""
We split the data to see how it will behave with new unseen data and
any data it is provided in the future.
If I test the data as I have trained it on. The results will obviously be 
great because the model is already aware of the data it was trained on.

"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n Training features:", X_train.shape)
print("\n Testing features:", X_test.shape)

# ===================================================================
# Step # 7 : Identify numerical and categorical columns
# ===================================================================
"""
The reason why identifying numerical and categorical columns is needed
because machine learning understands numbers.

Why we do numerical columns because the difference and variation in numbers
is too great so to bring them down to the same scale.

why we do categorical columns because the model needs numbers to understand.
Since they are text labels it wont be feasible so 1s and 0s is easily managable for the model.

"""

numeric_features = ["age", "income", "experience_years"]
categorical_features = ["education", "city"]

# ===================================================================
# Step # 8 : Numeric Pipeline
# ===================================================================
"""
The numeric pipeline will help in making the code look managable
prevent data leakage and most importantly we can use the same pipeline
for whenever we have data reproducibility

"""
numeric_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# ===================================================================
# Step # 9 : Categorical Pipeline
# ===================================================================

"""
The steps will be repeated for categorical pipeline and offer the same
benefits as for numeric pipeline

"""
categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# ===================================================================
# Step # 10 : Combine Preprocessing using ColumnTransformer
# ===================================================================
"""
We use ColumnTransformer because we have two seperate pipelines
which we need to combine and ColumnTransformer will connect them to the dataset.

"""
preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

# ===================================================================
# Step # 11 : Create Model Pipeline
# ===================================================================
"""
Creating a model pipeline is crucial because it connects the preprocessing
and the model into one object which makes the whole workflow run smoothly.

Benefits of Model pipeline is it prevents data leakage, easy to deploy,
fewer mistakes, and most importantly easy to tune and swap.
"""
# Creating a Logistic Regression Pipeline
lr_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))]
)

# Creating Random Forest Pipeline
rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# ===================================================================
# Step # 12 : Compare models using cross-validation
# ===================================================================
"""
Cross-validation is a way of testing a model by splitting the data into several parts, 
training on some and testing on the rest, repeating this so every part gets a turn
as the test set, and then averaging the scores.
"""
models = {"Logistic Regression": lr_pipeline, "Random Forest": rf_pipeline}
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
    cv_results[name] = scores.mean()

    print(f"\n{name}")
    print("F1 Scores:", scores)
    print("Mean CV F1:", round(scores.mean(), 4))


# ===================================================================
# Step # 13 : Select the best baseline candidate
# ===================================================================

best_model_name = max(cv_results, key=cv_results.get)
best_pipeline = models[best_model_name]

print("\nBest Baseline Candidate: ", best_model_name)


# ===================================================================
# Step # 14 : Define Hyperparameter Search Space
# ===================================================================
"""
The reason for using Hyperparameters is because one we find out which
model is performing and giving the best output we try to further squeeze
out the best performance

Once we test our data on default setting and find out the winner model.
we further try to squeeze out their best performance but tuning the hyperparameters

"""

if best_model_name == "Logistic Regression":
    param_grid = {  # Param grid is a dictionary that lists hyperparameters values you want GridSearchCV to try.
        "classifier__C": [
            0.01,
            0.1,
            1,
            10,
        ],  # Smaller value means stronger regularization, keeping the model simple.
        "classifier__class_weight": [None, "balanced"],
    }
else:
    param_grid = {  # Param grid is a dictionary that lists hyperparameters values you want GridSearchCV to try.
        "classifier__n_estimators": [100, 200],  # Number of Trees
        "classifier__max_depth": [
            None,
            10,
            20,
        ],  # No limit and how deep the Tree can grow
        "classifier__min_samples_leaf": [1, 2],  # No of leaves
    }

# ===================================================================
# Step # 15 : Tune Hyperparameter using GridSearchCV
# ===================================================================

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

print("\nBest cross-validation F1:")
print(round(grid_search.best_score_, 4))

# ===================================================================
# Step # 16 : Evaluate on Test Dataset
# ===================================================================
y_pred = best_model.predict(X_test)  # Model Prediction

y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nFinal Test Results are:")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))

print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))

print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 4))

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===================================================================
# Step # 17 : Save the complete trained pipeline
# ===================================================================

joblib.dump(best_model, "classification_pipeline.pkl")

print("\n Pipeline Saved Successfully.")
