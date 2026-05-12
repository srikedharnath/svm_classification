# svm_classification

# Insurance Charges Prediction using SVR

## Project Description

This project predicts medical insurance charges using Support Vector Regression (SVR).  
The prediction is based on several features such as age, gender, BMI, number of children, smoking status, and region.

The main objective of this project is to build a machine learning regression model that can accurately predict insurance charges.

---

# Dataset Information

The dataset contains 1338 rows and 7 columns.

## Features Used

| Column Name | Description |
|-------------|-------------|
| age | Age of the person |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of children |
| smoker | Smoking status |
| region | Residential region |
| charges | Insurance charges |

---

# Machine Learning Algorithm

- Support Vector Regression (SVR)

---

# Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Project Workflow

1. Load the dataset
2. Understand the data
3. Check null values
4. Encode categorical data
5. Feature scaling
6. Separate input and output variables
7. Split dataset into training and testing
8. Train the SVR model
9. Predict output values
10. Evaluate the model

---

# Data Preprocessing

## Label Encoding

Categorical columns encoded:
- sex
- smoker
- region

---

# Feature Scaling

StandardScaler is used because SVR is sensitive to feature ranges.

---

# SVR Hyperparameters Used

```python
SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)
