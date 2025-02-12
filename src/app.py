import streamlit as sl
from PIL import Image
import joblib
import numpy as np
import pandas as pd

model=joblib.load("titanic_model (1).pkl")
preprocessing = joblib.load("preprocessing.pkl")

sl.title("🚢 Titanic Survival Prediction App")
sl.text("This web app predicts whether a passenger survived the Titanic disaster or not.")
sl.image("b6f3d515-7874-403c-aac5-e8847e6884ae.webp")
sl.markdown("___")


sl.markdown("### Enter Passenger Details:")
Pclass=sl.radio("Pclass",options=[1,2,3])
sl.markdown("___")
Sex=sl.radio("Sex",options=("male","female"))
sl.markdown("___")
Embarked=sl.radio("Embarked",options=("S","C","Q"))
sl.markdown("___")
Age=sl.number_input("Age",min_value=0,max_value=100)
SibSp=sl.number_input("Sibilings",min_value=0,max_value=100)
Parch=sl.number_input("Parents",min_value=0,max_value=100)
Fare=sl.number_input("Fare",min_value=0.0,max_value=600.0)


features = pd.DataFrame([[Age, SibSp, Parch, Fare, Sex, Embarked, Pclass]],
                          columns=['Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Pclass'])

features_transformed = preprocessing.transform(features)

if sl.button("🚀predict"):
    prediction=model.predict(features_transformed)

    if prediction[0]==1:
        sl.success("🎉 Yeeh! The Passenger survived.")
    else:
        sl.error("💔 Unfortunality,The Passenger did not survive.")
