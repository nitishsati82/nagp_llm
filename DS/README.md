# Medical Virtual Assistant: Heart Attack Risk Prediction

## Background

The healthcare industry is increasingly investing in intelligent systems to enhance the efficiency of their services. One such application is the development of Medical Virtual Assistants. These assistants can help predict health risks based on patient data. In this project, we aim to predict the likelihood of a heart attack based on a dataset collected from a survey conducted by the US health organization on individuals aged between 30 to 80 years.

## Problem Objective

Using the provided dataset, the goal is to develop a Machine Learning (ML) model that can predict whether a person is at risk of having a heart attack.

## Domain

Health Services

## Dataset

The dataset is provided in the file `US_Heart_Patients.csv`.

## Instructions

### System setup 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report,accuracy_score
import warnings 
warnings.filterwarnings("ignore")
```

### 1. Load the Data

- Read the `US_Heart_Patients.csv` file from the folder into the program.
```python
data = pd.read_csv('US_Heart_Patients.csv', index_col=0)
```

### 2. Perform Exploratory Data Analysis (EDA)

Print the following information:
- The first 10 rows of the data.
```python
data.head(10)
```
- 5-point summary (minimum, first quartile (Q1), median, third quartile (Q3), and maximum).

```python
data.describe().T
```
- Information about the columns (data types).
```python
data.info()
```
- Number of outliers (extra points).
```python
count_outliers(data)
```
- Any missing values.
```python
data.isnull().sum()
```

- Correlation between variables.

```python
correlation_matrix = data.corr()
print("\nCorrelation matrix:")
correlation_matrix
```
- Distribution of the data.

Draw charts and graphs to visualize the above points where necessary.

### 3. Data Preprocessing

- Impute any missing values.
- Perform outlier treatment.
- Encode categorical features if needed.

### 4. Split the Dataset

- Split the data into 80% training and 20% test datasets.

### 5. Model Preparation and Evaluation

Run the following steps for Naïve Bayes, and Decision Tree:
- Train the model and predict the output for both train and test data.
- Calculate F1 score.

Pick and explain the best model out of the two and explain its confusion matrix and classification report.

---

*Use Random Seed = 42 everywhere*

### Github code link: https://github.com/nitishsati82/nagp_llm/tree/main/DS