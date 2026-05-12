# Iris Flower Classification using Support Vector Machine (SVM)

## Overview

This project uses Support Vector Machine (SVM) to classify iris flowers into different species based on flower measurements.

The model predicts the flower species using:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

This project demonstrates:
- Data preprocessing
- Feature Scaling
- Train-Test Split
- SVM Model Training
- Model Evaluation

---

# Dataset Information

The Iris dataset contains 150 rows and 5 columns.

## Features

| Column Name | Description |
|-------------|-------------|
| SepalLengthCm | Length of sepal |
| SepalWidthCm | Width of sepal |
| PetalLengthCm | Length of petal |
| PetalWidthCm | Width of petal |
| Species | Flower species |

---

# Target Classes

The model predicts one of the following flower species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

# Problem Type

This is a:
# Classification Problem

because the target variable contains categories/classes.

---

# Machine Learning Algorithm Used

- Support Vector Machine (SVM)

---

# Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Project Workflow

1. Load Dataset
2. Understand Dataset
3. Check Null Values
4. Remove Unnecessary Columns
5. Separate Features and Target
6. Train-Test Split
7. Feature Scaling
8. Train SVM Model
9. Predict Output
10. Evaluate Model

---

# Data Preprocessing

## Removed Column

- Id column was removed because it is not useful for prediction.

---

# Feature Scaling

StandardScaler was used because SVM is sensitive to feature ranges.

---

# SVM Hyperparameters Used

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)
