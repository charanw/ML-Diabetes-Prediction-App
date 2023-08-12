import joblib
import streamlit as st

# Set configuration settings for the page
st.set_page_config(
    page_title="Prediction Tool - WGU Capstone",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state='auto'
)


# Function to load and cache the model and scaler
@st.cache_data
def initialize():
    model = joblib.load('diabetes_model.save')
    scaler = joblib.load("diabetes_scaler.save")
    return model, scaler


model, scaler = initialize()

# Begin page content
st.title("❔ Prediction Tool")
st.divider()
st.header('Input Member Data')
st.divider()

# Form for inputting member data to generate predictions
form = st.form('Member_Info')
p_sex = form.radio(label='Sex', options=['Male', 'Female', 'Other'], key="Gender", horizontal=True)
p_age = form.number_input(key='Age', label='Age', min_value=1, max_value=120, value=18)
p_hyp = form.selectbox(key='Hyp', label='Does member have hypertension?', options=['No', 'Yes'])
p_heart = form.selectbox(key='Heart', label='Does member have heart disease?', options=['No', 'Yes'])
p_smoke = form.selectbox(key='Smoke', label='Does member smoke?',
                         options=['Has Never Smoked', 'Formerly Smoked', 'Currently Smokes'])
p_bmi = form.number_input(key='BMI', label='BMI', min_value=0.0, value=25.0, step=1.0)
p_a1c = form.number_input(key='A1C', label='A1C', min_value=0.0, value=5.7, step=1.0)
p_gluc = form.number_input(key='Gluc', label='Blood Glucose Level', min_value=0.0, value=90.0, step=10.0)
submitted = form.form_submit_button('Calculate')
st.divider()

result_container = st.container()
# When the form is submitted, prepare the inputs and call the model's predict function to generate a prediction
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

    if p_smoke == 'Has Never Smoked':
        Has_Never_Smoked = 1
    elif p_smoke == 'Formerly Smoked':
        Formerly_Smoked = 1
    elif p_smoke == 'Currently Smokes':
        Currently_Smokes = 1

    # Standardize the data with the loaded scaler before passing it to the model
    input_data = scaler.transform([[p_age, male, female, other, Has_Never_Smoked, Formerly_Smoked, Currently_Smokes,
                                    hyp, heart, p_bmi, p_a1c, p_gluc]])
    prediction = model.predict(input_data)
    predict_probability = model.predict_proba(input_data)

    # Display the results
    if prediction[0] == 1:
        result_container.title('Results')
        result_container.error(
            'Member is positive for diabetes. Probability: {}%'.format(round(predict_probability[0][1] * 100, 3)),
            icon="⚠️")
        result_container.divider()
    else:
        result_container.title('Results')
        result_container.success(
            'Member is negative for diabetes. Probability: {}%'.format(round(predict_probability[0][0] * 100, 3)),
            icon="✅")
        result_container.divider()

st.write('Created by Charan Williams')
st.write('August 10th, 2023')
st.caption('Disclaimer: This application is for educational and demonstration purposes only and is not intended to provide actual '
         'medical advice. Always consult a licensed healthcare provider if you have any health concerns.')
