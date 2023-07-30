import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import streamlit as st
import joblib

st.set_page_config(
    page_title="Prediction Tool - WGU Capstone",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state='auto'
)

@st.cache_data
def initialize():
    model = joblib.load('diabetes_model.save')
    scaler = joblib.load("diabetes_scaler.save")
    return model, scaler

model, scaler = initialize()

st.title("❔ Prediction Tool")
st.divider()
container = st.container()
st.header('Input Patient Data')
st.divider()
form = st.form('Patient_Info')
p_sex = form.radio(label='Sex', options=['Male', 'Female', 'Other'], key="Gender", horizontal=True)
p_age = form.number_input(key= 'Age', label='Age', min_value=1, max_value=120, value=18)
p_hyp = form.selectbox(key='Hyp', label='Does patient have hypertension?', options=['No','Yes'])
p_heart = form.selectbox(key='Heart', label='Does patient have heart disease?', options=['No','Yes'])
p_smoke = form.selectbox(key='Smoke', label='Does patient smoke?', options=['Has Never Smoked', 'Formerly Smoked', 'Currently Smokes'])
p_bmi = form.number_input(key= 'BMI', label='BMI', min_value=0.0, value=25.0, step=1.0)
p_a1c = form.number_input(key= 'A1C', label='A1C', min_value=0.0, value=5.7, step=1.0)
p_gluc = form.number_input(key= 'Gluc', label='Blood Glucose Level', min_value=0.0, value=90.0, step=10.0)
submitted = form.form_submit_button('Calculate')

if submitted:
    male, female, other, hyp, heart, Has_Never_Smoked, Formerly_Smoked, Currently_Smokes = 0, 0, 0, 0, 0, 0, 0, 0
    if p_sex == 'Male':
        male = 1
    elif p_sex == 'Female':
        female = 1
    else:
        other = 1

    if p_hyp == 'Yes':
        hyp = 1

    if p_heart == 'Yes':
        heart = 1

    elif p_smoke == 'Has Never Smoked':
        Has_Never_Smoked = 1
    elif p_smoke == 'Formerly Smoked':
        Formerly_Smoked = 1
    elif p_smoke == 'Currently Smokes':
        Currently_Smokes = 1

    input_data = scaler.transform([[p_age, male, female, other, Has_Never_Smoked, Formerly_Smoked, Currently_Smokes, hyp, heart, p_bmi, p_a1c, p_gluc]])
    print(input_data)
    prediction = model.predict(input_data)
    print(prediction)
    predict_probability = model.predict_proba(input_data)
    print(predict_probability)

    if prediction[0] == 1:
        container.title('Results')
        container.error('Patient is positive for diabetes. Probability: {}%'.format(round(predict_probability[0][1]*100, 3)), icon="⚠️")
        container.divider()
    else:
        container.title('Results')
        container.success('Patient is negative for diabetes. Probability: {}%'.format(round(predict_probability[0][0] * 100, 3)), icon="✅")
        container.divider()

st.divider()
