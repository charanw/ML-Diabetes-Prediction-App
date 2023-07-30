import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Home - WGU Capstone",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state='collapsed'
)


# Function to load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv", low_memory=False)
    return df

# Save the data as a state variable so that it can be accessed by other pages
if 'df' not in st.session_state:
    st.session_state.df = load_data()

# Save the data as a variable for use on this page
df_src = st.session_state.df

# Begin page content
st.title("⚕️AI Diabetes Prediction")
st.divider()
st.header("Are you at risk?")
col1, col2 = st.columns([.5, .5])
with col1:
    st_lottie("https://lottie.host/38df451d-8843-4a72-978e-436e911ebb5c/LoSDYJIVwo.json", height=300)
with col2:
    "According to the blah blah blah, millions of americans are at risk of developing diabetes over the course of their lifetime. Diabetes is a serious disease that will seriously impact the quality of your life! Find out your risk of developiong diabetes so you can be informed and make changes now!"
st.divider()
st.header("Data")
col3, col4 = st.columns([.5,.5])
with col3:
    "This tool was trained on a dataset provided by Kaggle.com. It contains anonymized health information on patients " \
    "positive and negative for diabetes including metrics such as gender, age, BMI, A1C, blood glucose measurement, and " \
    "whether the patient has hypertension or heart disease. "
with col4:
    st_lottie("https://lottie.host/d8f3f141-4582-4d20-a218-0aefd66e1424/YNDRN0F5y6.json", height=300)
st.divider()
st.header("Get Started")
st.subheader('Thank you for checking out my application.')
st.write('I suggest you begin by clicking one of the buttons below. You can also use the left sidebar for easy navigation.')
col5, col6, col7, = st.columns(3)
with col5:
    st.markdown('<a target="_self" href=/Prediction_Tool><button id="cta_button_1" style="background-color:#58A4B0; border: none; text-decoration: none; padding: 20px; width: 200px; border-radius: 5px;">Prediction Tool</button></a>', unsafe_allow_html=True)
with col6:
    st.markdown('<a target="_self" href=/Query_Dataset><button id="cta_button_1" style="background-color:#58A4B0; border: none; text-decoration: none; padding: 20px; width: 200px; border-radius: 5px;">Query Dataset</button></a>', unsafe_allow_html=True)
with col7:
    st.markdown('<a target="_self" href=/Dashboard><button style="background-color:#58A4B0; border: none; text-decoration: none; padding: 20px; width: 200px; border-radius: 5px;">Dashboard</button></a>', unsafe_allow_html=True)
st.divider()
st.write('Created by Charan Williams July, 2023')