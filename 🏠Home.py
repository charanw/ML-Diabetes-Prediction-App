import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie

# Set configuration settings for the page
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
st.title("⚕️ML Diabetes Prediction")
st.divider()
st.header("Catch Diabetes Early")
col1, col2 = st.columns([.5, .5])
with col1:
    st_lottie("https://lottie.host/38df451d-8843-4a72-978e-436e911ebb5c/LoSDYJIVwo.json", height=300)
with col2:
    "According to Cecelia Health (2019) persons with diabetes cost payer organizations 2.3 times more than " \
    "non-diabetic members. In addition, Cecelia Health also states that 1 in 7 dollars healthcare dollars is spent " \
    "treating diabetes and related complications."
"This application addresses the rising costs of claims from diabetic " \
    "members by using a logistic regression machine learning algorithm to predict a diabetes diagnosis, which will " \
    "enable the payer organization to intervene earlier. Early intervention will improve member health outcomes and " \
    "save the payer organization sizable costs."
st.divider()
st.header("Data")
col3, col4 = st.columns([.5, .5])
with col3:
    "This tool was trained on a dataset provided by Kaggle.com. It contains anonymized health information on patients " \
    "positive and negative for diabetes including metrics such as gender, age, BMI, A1C, blood glucose measurement, " \
    "and whether the patient has hypertension or heart disease. "
with col4:
    st_lottie("https://lottie.host/d8f3f141-4582-4d20-a218-0aefd66e1424/YNDRN0F5y6.json", height=300)
st.divider()
st.header("Get Started")
st.subheader('Thank you for checking out my application.')
st.write(
    'I suggest you begin by clicking one of the buttons below. You can also use the left sidebar for easy navigation.')
col5, col6, col7, = st.columns(3)
with col5:
    st.markdown(
        '<a target="_self" href=/Prediction_Tool><button id="cta_button_1" style="background-color:#58A4B0; border: '
        'none; text-decoration: none; padding: 20px; width: 200px; border-radius: 5px;">Prediction Tool</button></a>',
        unsafe_allow_html=True)
with col6:
    st.markdown(
        '<a target="_self" href=/Query_Dataset><button id="cta_button_1" style="background-color:#58A4B0; border: '
        'none; text-decoration: none; padding: 20px; width: 200px; border-radius: 5px;">Query Dataset</button></a>',
        unsafe_allow_html=True)
with col7:
    st.markdown(
        '<a target="_self" href=/Dashboard><button style="background-color:#58A4B0; border: none; text-decoration: '
        'none; padding: 20px; width: 200px; border-radius: 5px;">Dashboard</button></a>',
        unsafe_allow_html=True)
st.divider()
st.header('References')
st.markdown('Cecelia Health. (2019, April 9). <i> The true cost of diabetes to health insurance organizations</i> - Cecelia Health. Retrieved August 10, 2023, from https://www.ceceliahealth.com/industry-insights/the-true-cost-of-diabetes-to-health-insurance-organizations/', unsafe_allow_html=True)
st.markdown('Mustafa, M. (2023). <i>Diabetes prediction dataset: A Comprehensive Dataset for Predicting Diabetes with Medical & Demographic Data</i> (Version 1) [Dataset]. https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset', unsafe_allow_html=True)
st.write('Homepage graphics from lottiefiles.com')
st.divider()
st.write('Created by Charan Williams August, 2023')
st.caption('Disclaimer: This application is for educational and demonstration purposes only and is not intended to provide actual '
           'medical advice. Always consult a licensed healthcare provider if you have any health concerns.')