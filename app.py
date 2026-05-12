import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


st.title("Iris Flower Classification using SVM")

st.write("Enter flower measurements below:")


df = pd.read_csv("Iris.csv")


df = df.drop("Id", axis=1)


X = df.drop("Species", axis=1)
y = df["Species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    degree=3,
    probability=True
)

model.fit(X_train, y_train)


sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")


if st.button("Predict"):

  
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=X.columns
    )

   
    input_data = scaler.transform(input_data)

   
    prediction = model.predict(input_data)

    
    st.success(f"Predicted Flower Species: {prediction[0]}")
