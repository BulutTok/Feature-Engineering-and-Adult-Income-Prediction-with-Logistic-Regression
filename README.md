# Feature Engineering and Adult Income Prediction with Logistic Regression

This repository provides an example workflow for preprocessing, feature engineering, and building a Logistic Regression model using the UCI Adult dataset. The project demonstrates how to handle categorical variables, perform one-hot encoding, scale numerical features, and evaluate a machine learning model with Python libraries such as `mglearn`, `pandas`, `numpy`, and `scikit-learn`.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Code Walkthrough](#code-walkthrough)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project uses the UCI Adult dataset to predict whether a person earns over \$50K per year. The workflow includes:

- Loading and preparing the dataset.
- Creating dummy variables for categorical features.
- Splitting the data into training and testing sets.
- Training a Logistic Regression model.
- Scaling features using `StandardScaler`.
- Using a `ColumnTransformer` to combine scaling and one-hot encoding in a single pipeline.

---

## Installation

Ensure you have Python 3.6+ installed. Install the necessary packages using pip:

```bash
pip install mglearn numpy pandas scikit-learn
```

---

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/adult-income-prediction.git
   cd adult-income-prediction
   ```

2. **Open the project in your favorite IDE or Jupyter Notebook.**

3. **Run the script or notebook:**

   - If using a script:
     ```bash
     python your_script.py
     ```
   - If using Jupyter Notebook, open the notebook file and run each cell sequentially.

---

## Project Structure

```
.
├── README.md            # This file
├── your_script.py       # Main Python script (or Notebook.ipynb)
└── requirements.txt     # List of required packages
```

---

## Code Walkthrough

### 1. Installation and Imports

The project starts by installing and importing necessary libraries:

```python
!pip install mglearn
import os
import mglearn
import numpy as np
import pandas as pd
```

### 2. Data Loading and Preprocessing

The UCI Adult dataset is loaded using `mglearn.datasets.DATA_PATH`, and column names are provided manually since the dataset file lacks headers:

```python
adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(
    adult_path, header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation', 'income']]
display(data.head())
```

### 3. Handling Categorical Variables

Dummy variables are created for categorical features using `pd.get_dummies`:

```python
data_dummies = pd.get_dummies(data)
print(data_dummies.columns)
```

### 4. Feature Extraction, Splitting, and Model Training

Features and the target variable are separated, and the data is split into training and testing sets. A Logistic Regression model is then trained:

```python
features = data_dummies.loc[:,"age":"occupation_ Transport-moving"]
X = features.values
y = data_dummies['income_ >50K']
print(X.shape, y.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

log = LogisticRegression()
log.fit(X_train, y_train)
print("Training score:", log.score(X_train, y_train))

y_pred = log.predict(X_test)
print("Test score:", log.score(X_test, y_pred))
```

### 5. Feature Scaling and Advanced Encoding

Numerical features are scaled using `StandardScaler`, and a `ColumnTransformer` is used to combine scaling with one-hot encoding for categorical features:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Scale numerical columns
columns_to_scale = ['age', 'hours-per-week']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['age', 'hours-per-week']])
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
columns_categorical = ['workclass', 'education', 'gender', 'occupation', 'income']
scaled_df[columns_categorical] = data[columns_categorical]

print("Original DataFrame:")
print(data)
print("\nScaled DataFrame:")
print(scaled_df)

# ColumnTransformer for combined preprocessing
ct = ColumnTransformer([
    ('scaling', StandardScaler(), ['age', 'hours-per-week']),
    ('onehot', OneHotEncoder(sparse_output=False), ['workclass', 'education', 'gender', 'occupation'])
])

features = data.loc[:, "age":"occupation"]
X = features
y = data['income']
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

ct.fit(X_train)
X_train_trans = ct.transform(X_train)
print("Transformed training data shape:", X_train_trans.shape)

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train_trans, y_train)
print("Training score after transformation:", log.score(X_train_trans, y_train))

X_test_trans = ct.transform(X_test)
y_pred = log.predict(X_test_trans)
print("Test score after transformation:", log.score(X_test_trans, y_pred))
```

---



## Acknowledgments

- **mglearn**: For providing access to useful datasets and machine learning utilities.
- **Scikit-learn**: For the machine learning tools and preprocessing techniques.
- **UCI Machine Learning Repository**: For the Adult dataset used in this project.

