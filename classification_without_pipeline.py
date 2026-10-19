# ==================================================================
"""
In this project I will working on the complete workflow of Machine learning.
But I will not be using pipelines just to have the feel on how extensive
and time consuming our work becomes especially when dealing with
preprocessing department scaler, imputer, and onehotencoding.
"""
# =================================================================

# ===================================================================
# Step # 1: Import libraries
# ===================================================================
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

# ===================================================================
# Step # 2: Creating Dataset
# ===================================================================
"""
I will now create an artifical dataset with a 1000 samples. But no pipelines
will be introduced in this project.
"""

np.random.seed(42)
n_samples = 1000

# Introducing artifical data
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

# Convert the data into a Dataframe
df = pd.DataFrame(data)

# Print statement to check if the Dataframe has been created successfully.

print(df.head())
# print the first five rows of the Dataframe

print(df.isnull().sum())
# To check if there are any missing values in the dataset
# Since there are no missing values I will introduce them in the dataset.

# ===================================================================
# Step # 3: Introducing Missing Values
# ===================================================================
"""
I won't including missing values in all of the columns but only in 2
columns education and experience.
"""
df.loc[df.sample(frac=0.1, random_state=1).index, "income"] = np.nan
# Introducing missing values in Income Column

df.loc[df.sample(frac=0.09, random_state=2).index, "education"] = np.nan
# Introducing missing values in Experience Years

# ===================================================================
# Step # 4: Checking Missing Values
# ===================================================================
"""
Print statement offers the best solution to check the changes which have been made.
"""
print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nPercentage of Missing Values:")
print(df.isna().mean() * 100)

print("The First Five Rows of the Dataset")
print(df.head())

print("The Last Five Rows of the Dataset")
print(df.tail())

print("\nTarget Distribution:")
print(df["target"].value_counts())

# ===================================================================
# Step # 5 : Seperate Features (X) and Target (y)
# ===================================================================
"""
In this step I will separating the dataframe into features and targets(labels).
'X' is our feature and holds the inputs while 'y' is our target
on which predicting will be done.
"""
X = df.drop(columns=["target"])
# The target column is initially dropped because it will be part of
# y which will be used for predicting
y = df["target"]

print("\nFeatures")
print(X.head())  # Print the first five rows of the dataset
print(X.tail())  # Print the last five rows of the dataset

print("\nTarget")
print(y.head())  # Print the first five rows of the dataset
print(y.tail())  # Print the last five rows of the dataset

# ===================================================================
# Step # 6 : Splitting into Training and Testing
# ===================================================================
"""
The data needs to be splt in training and testing because we need to analyze 
how the model will behave on new unseen data.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n Training Features:", X_train.shape)
print("\n Testing Features:", X_test.shape)

# ===================================================================
# Step # 7 : Identify numerical and categorical columns
# ===================================================================

# First identify the numeric columns in the dataset.
numeric_features = ["age", "income", "experience_years"]

# Secondly identify the categorical features in the dataset.
categorical_features = ["education", "city"]

# ===================================================================
# Step # 8 : Handle the Missing Numerical Columns
# ===================================================================
"""
There are two options either I apply SimpleImputer to only the columns which
have missing data in their columns.
Or
I can apply SimpleImputer to all of the columns just to be on the safe side
SimpleImputer will leave the columns unchangd which does not have missing values.

Median as a strategy is used the fill the missing numerical values with the middle
value of the column. This strategy is especially useful when the data contains unusally
high or low values called outliers

Outliers is a datavalue which is far away from most other values in a dataset.
"""
numeric_imputer = SimpleImputer(strategy="median")

X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])

# ===================================================================
# Step # 9 : Handle the Missing Categorical Columns
# ===================================================================
"""
for categorical columns which are non-numeric there are usually two strategies
most_frequent : which uses the most common value in the column
constant : fill the missing value with specific value
"""

categorical_imputer = SimpleImputer(strategy="most_frequent")

X_train[categorical_features] = categorical_imputer.fit_transform(
    X_train[categorical_features]
)
X_test[categorical_features] = categorical_imputer.transform(
    X_test[categorical_features]
)

# ===================================================================
# Step # 10 : Encode the first Missing Value : Education (Natural Order)
# ===================================================================
"""
Encoding will help to covert the non-numerical (text) data into machine readable
numerical value.
A dictionary is created which will help create key and value each key in the case
of education level is given a numerical value.

The .map(education_order) looks into the key value of the dictionary and 
replaces it with it numerical value.

Then the original column is dropped and replaced with the new education numerical column 
"""
education_order = {"High School": 1, "Bachelor": 2, "Master": 3, "PHD": 4}

X_train["education_encoded"] = X_train["education"].map(education_order)
X_test["education_encoded"] = X_test["education"].map(education_order)


X_train.drop(columns=["education"], inplace=True)
X_test.drop(columns=["education"], inplace=True)

# ===================================================================
# Step # 11 : OneHot Encode : City (No Natural Order)
# ===================================================================
"""
OneHot Encoding is used on no-natural order / nominal data.
"""
X_train = pd.get_dummies(X_train, columns=["city"], dtype=int)

X_test = pd.get_dummies(X_test, columns=["city"], dtype=int)

# A small test to check if both the Training and Testing columns match
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ===================================================================
# Step # 12 : Feature Engineering
# ===================================================================
"""
One of the most important aspects in Machine Learning is Feature Engineering
Feature engineering is the process of creating, transforming, 
or selecting features to help a Machine Learning model learn more effectively.
"""
for data_part in [X_train, X_test]:
    # Experience relative to age
    data_part["experience_ration"] = data_part["experience_years"] / data_part[
        "age"
    ].replace(0, np.nan)

    # Income divided by experience plus one
    data_part["income_per_experience"] = data_part["income"] / (
        data_part["experience_years"] + 1
    )

    # 1= Masters or PHD = 0 otherwise
    data_part["high_education"] = (data_part["education_encoded"] >= 3).astype(int)

# ===================================================================
# Step # 13 : Handling invalid values in Featured Engineered
# ===================================================================
"""
The steps are perfomed to insure that the new columns which were created
from feature engineering have no missing values in them
"""

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

feature_imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(
    feature_imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)

X_test = pd.DataFrame(
    feature_imputer.transform(X_test), columns=X_test.columns, index=X_test.index
)

# ===================================================================
# Step # 14 : Scaler the Features
# ===================================================================
"""
Scaling is done so bring large numerical values onto a similar scale.
Having it on a similar scale helps will prevent it from dominating the model.

"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================================================================
# Step # 15 : Checking the final results
# ===================================================================
print("\nTraining data shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

print("Training Data Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

print("\nMissing Values in Training Features:", np.isnan(X_train_scaled).sum())
print("\n Missing Values in Testing Features:", np.isnan(X_test_scaled).sum())

# ===================================================================
# Step # 16 : Saving The Processed Data
# ===================================================================
np.save("X_train_scaled.npy", X_train_scaled)
np.save("X_test_scaled.npy", X_test_scaled)

np.save("y_train.npy", y_train.to_numpy())
np.save("y_test.npy", y_test.to_numpy())

print("\nPreprocessing without Pipeline Completed Successfully!")

# ===================================================================
# Step # 17 : Deplying Model
# ===================================================================
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
random_forest = RandomForestClassifier(n_estimators=200, random_state=42)


# ===================================================================
# Step # 18 : Model Fit
# ===================================================================
"""
Model Fit is done to train the model
"""
logistic_model.fit(X_train_scaled, y_train)
random_forest.fit(X_train_scaled, y_train)

# ===================================================================
# Step # 19 : Model Evaluation
# ===================================================================
"""
Since there are two models which are being deployed
"""
models = {"Logistic Regression": logistic_model, "Random Forest": random_forest}
# In a Dictionary a key and value is seperated by a 'colon' and a pair by a 'comma'

results = []

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob),
        }
    )

    print(f"\n{name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# The print statement below will showcase both the models metric results
print("=" * 65)
print(pd.DataFrame(results).set_index("Model").round(3))
print("=" * 65)
