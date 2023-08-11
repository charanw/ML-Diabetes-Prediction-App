import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Set configuration settings for the page
st.set_page_config(
    page_title="Dashboard - WGU Capstone",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state='auto'
)


# Function to load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv", low_memory=False)
    return df


# Function to load and cache the confusion matrix generated when testing the model
@st.cache_data
def initialize():
    confusion_matrix = joblib.load("diabetes_confusion_matrix.save")
    return confusion_matrix


# Save the data as a state variable so that it can be accessed by other pages
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df_src = st.session_state.df

# Begin Page Content
st.title("📊 Dashboard")
st.divider()

st.header("Model Performance")

# Create and display accuracy bar chart
confusion_matrix = initialize()
confusion_table = go.Figure(
    data=[go.Bar(
        x=["False Negative", "True Negative", "True Positive", "False Positive"],
        y=[confusion_matrix[1][0], confusion_matrix[0][0], confusion_matrix[1][1], confusion_matrix[0][1]])])

st.plotly_chart(confusion_table, use_container_width=True, height=300)

# Load and display the training and testing scores of the model
scores = joblib.load("training_and_testing_scores.save")
training_score = scores[0]
testing_score = scores[1]
st.subheader("Training Set Score = {}%".format(round(training_score, 3) * 100))
st.subheader("Testing Set Score = {}%".format(round(testing_score, 3) * 100))
st.divider()

# Create 2 column layout
st.header('Gender Breakdown')
col1, col2 = st.columns(2, gap='large')

# Create gender breakdown dataframe and pie chart
with col1:
    diabetes_negative_gender_totals = df_src.query('diabetes == 0')['gender'].value_counts()
    st.subheader('Diabetes Negative')
    st.plotly_chart(
        px.pie(data_frame=diabetes_negative_gender_totals, names=diabetes_negative_gender_totals.index, values='gender', color=diabetes_negative_gender_totals.index,
               color_discrete_map={'Male': '#57799E', 'Female': '#DAA49A', 'Other': '#41818B'},
               hover_name=diabetes_negative_gender_totals.index), use_container_width=True, height=200)

with col2:
    diabetes_positive_gender_totals = df_src.query('diabetes == 1')['gender'].value_counts()
    st.subheader('Diabetes Positive')
    st.plotly_chart(
        px.pie(data_frame=diabetes_positive_gender_totals, names=diabetes_positive_gender_totals.index, values='gender', color=diabetes_positive_gender_totals.index,
               color_discrete_map={'Male': '#57799E', 'Female': '#DAA49A', 'Other': '#41818B'},
               hover_name=diabetes_positive_gender_totals.index), use_container_width=True, height=200)

st.divider()
st.header("BMI, A1C, and Age Distribution")
# Create dataframe with all bmi data and only bmi data of those who have diabetes
diabetes_negative_bmi_data = go.Histogram(x=df_src.query('diabetes == 0')['bmi'], marker=dict(color='#6E8EAF'),
                                          nbinsx=20, name="Diabetes Negative")
diabetes_positive_bmi_data = go.Histogram(x=df_src.query('diabetes == 1')['bmi'],
                                          marker=dict(color='#F52D00', opacity=.4), nbinsx=20,
                                          name="Diabetes Positive")
# Create histogram to plot bmi distribution
bmi_hist = make_subplots(specs=[[{"secondary_y": True}]])
bmi_hist.add_trace(diabetes_negative_bmi_data, secondary_y=False)
bmi_hist.add_trace(diabetes_positive_bmi_data, secondary_y=True)
bmi_hist.update_layout(xaxis_title='BMI', yaxis_title='Count', legend_title='Key')

st.subheader('BMI Distribution')
st.plotly_chart(bmi_hist, use_container_width=True, height=300)

# Create dataframe with all A1C data and only A1C data of those who have diabetes
diabetes_negative_a1c_data = go.Histogram(x=df_src.query('diabetes == 0')['HbA1c_level'], marker=dict(color='#6E8EAF'),
                                          nbinsx=10, name="Diabetes Negative")
diabetes_positive_a1c_data = go.Histogram(x=df_src.query('diabetes == 1')['HbA1c_level'],
                                          marker=dict(color='#F52D00', opacity=.4), nbinsx=10, name="Diabetes Positive")

# Create histogram to plot A1C distribution
a1c_hist = make_subplots(specs=[[{"secondary_y": True}]])
a1c_hist.add_trace(diabetes_negative_a1c_data, secondary_y=False)
a1c_hist.add_trace(diabetes_positive_a1c_data, secondary_y=True)
a1c_hist.update_layout(xaxis_title='A1C', yaxis_title='Count', legend_title='Key')

st.subheader('A1C Distribution')
st.plotly_chart(a1c_hist, use_container_width=True, height=300)

# Create dataframe with all age data and only age data of those who have diabetes
diabetes_negative_age_data = go.Histogram(x=df_src.query('diabetes == 0')['age'], marker=dict(color='#6E8EAF'),
                                          nbinsx=100, name="Diabetes Negative")
diabetes_positive_age_data = go.Histogram(x=df_src.query('diabetes == 1')['age'],
                                          marker=dict(color='#F52D00', opacity=.4), nbinsx=100,
                                          name="Diabetes Positive")

# Create histogram to plot age distribution
age_hist = make_subplots(specs=[[{"secondary_y": True}]])
age_hist.add_trace(diabetes_negative_age_data, secondary_y=False)
age_hist.add_trace(diabetes_positive_age_data, secondary_y=True)
age_hist.update_layout(xaxis_title='Age', yaxis_title='Count', legend_title='Key')

st.subheader('Age Distribution')
st.plotly_chart(age_hist, use_container_width=True, height=300)

st.divider()
st.header('Data Analysis')

# Create another 2 column layout
col3, col4 = st.columns(2, gap='small')
with col3:
    # Calculate and display the data completion rate for each column in a table
    st.subheader('Data Completion')
    st.table((df_src.count() / len(df_src)) * 100)

with col4:
    # Calculate and display the counts of relevant values in a table
    st.subheader('Counts')
    st.table(df_src['hypertension'].value_counts())
    st.table(df_src['heart_disease'].value_counts())
    st.table(df_src['diabetes'].value_counts())
    st.table(df_src['smoking_history'].value_counts())

# Create and display a stats table for relevant values of both diabetes negative and positive records,
# including mean, median, mode, min, max, and standard deviation
negative_stats_table = pd.DataFrame(index=['Diabetes Negative', ],
                                    columns=['Mean', 'Median', 'Mode', 'Min', 'Max', 'Standard Deviation'])
positive_stats_table = pd.DataFrame(index=['Diabetes Positive', ],
                                    columns=['Mean', 'Median', 'Mode', 'Min', 'Max', 'Standard Deviation'])

st.header('Diabetes Negative Stats')
negative_stats_table = pd.DataFrame(data={'Mean': [df_src.query('diabetes == 0')['age'].mean(),
                                                   df_src.query('diabetes == 0')['bmi'].mean(),
                                                   df_src.query('diabetes== 0')['HbA1c_level'].mean(),
                                                   df_src.query('diabetes== 0')['blood_glucose_level'].mean()
                                                   ],
                                          'Median': [df_src.query('diabetes == 0')['age'].median(),
                                                     df_src.query('diabetes == 0')['bmi'].median(),
                                                     df_src.query('diabetes== 0')['HbA1c_level'].median(),
                                                     df_src.query('diabetes== 0')['blood_glucose_level'].median()],
                                          'Mode': [df_src.query('diabetes == 0')['age'].mode()[0],
                                                   df_src.query('diabetes == 0')['bmi'].mode()[0],
                                                   df_src.query('diabetes== 0')['HbA1c_level'].mode()[0],
                                                   df_src.query('diabetes== 0')['blood_glucose_level'].mode()[0]],
                                          'Min': [df_src.query('diabetes == 0')['age'].min(),
                                                  df_src.query('diabetes == 0')['bmi'].min(),
                                                  df_src.query('diabetes== 0')['HbA1c_level'].min(),
                                                  df_src.query('diabetes== 0')['blood_glucose_level'].min()],
                                          'Max': [df_src.query('diabetes == 0')['age'].max(),
                                                  df_src.query('diabetes == 0')['bmi'].max(),
                                                  df_src.query('diabetes== 0')['HbA1c_level'].max(),
                                                  df_src.query('diabetes== 0')['blood_glucose_level'].max()],
                                          'Standard Deviation': [df_src.query('diabetes == 1')['age'].std(),
                                                                 df_src.query('diabetes == 0')['bmi'].std(),
                                                                 df_src.query('diabetes== 0')['HbA1c_level'].std(),
                                                                 df_src.query('diabetes== 0')[
                                                                     'blood_glucose_level'].std()]},
                                    index=['Age', 'BMI', 'A1C', 'Blood Glucose'])
st.table(negative_stats_table)

st.header('Diabetes Positive Stats')
positive_stats_table = pd.DataFrame(data={'Mean': [df_src.query('diabetes == 1')['age'].mean(),
                                                   df_src.query('diabetes == 1')['bmi'].mean(),
                                                   df_src.query('diabetes== 1')['HbA1c_level'].mean(),
                                                   df_src.query('diabetes== 1')['blood_glucose_level'].mean()
                                                   ],
                                          'Median': [df_src.query('diabetes == 1')['age'].median(),
                                                     df_src.query('diabetes == 1')['bmi'].median(),
                                                     df_src.query('diabetes== 1')['HbA1c_level'].median(),
                                                     df_src.query('diabetes== 1')['blood_glucose_level'].median()],
                                          'Mode': [df_src.query('diabetes == 1')['age'].mode()[0],
                                                   df_src.query('diabetes == 1')['bmi'].mode()[0],
                                                   df_src.query('diabetes== 1')['HbA1c_level'].mode()[0],
                                                   df_src.query('diabetes== 1')['blood_glucose_level'].mode()[0]],
                                          'Min': [df_src.query('diabetes == 1')['age'].min(),
                                                  df_src.query('diabetes == 1')['bmi'].min(),
                                                  df_src.query('diabetes== 1')['HbA1c_level'].min(),
                                                  df_src.query('diabetes== 1')['blood_glucose_level'].min()],
                                          'Max': [df_src.query('diabetes == 1')['age'].max(),
                                                  df_src.query('diabetes == 1')['bmi'].max(),
                                                  df_src.query('diabetes== 1')['HbA1c_level'].max(),
                                                  df_src.query('diabetes== 1')['blood_glucose_level'].max()],
                                          'Standard Deviation': [df_src.query('diabetes == 1')['age'].std(),
                                                                 df_src.query('diabetes == 1')['bmi'].std(),
                                                                 df_src.query('diabetes== 1')['HbA1c_level'].std(),
                                                                 df_src.query('diabetes== 1')[
                                                                     'blood_glucose_level'].std()]},
                                    index=['Age', 'BMI', 'A1C', 'Blood Glucose'])
st.table(positive_stats_table)

# Calculate and display a table of the percent of records with comorbidities for both diabetes positive and negative
# records
st.header('Percent with Comorbidities')
percentage_stats_table = pd.DataFrame(data={'% With Heart Disease': [
    len(df_src.query('diabetes == 0 and heart_disease==1')) / len(df_src.query('diabetes==0')) * 100,
    len(df_src.query('diabetes == 1 and heart_disease==1')) / len(df_src.query('diabetes==1')) * 100],
    '% With Hypertension': [
        len(df_src.query('diabetes == 0 and hypertension==1')) / len(
            df_src.query('diabetes==0')) * 100,
        len(df_src.query('diabetes == 1 and hypertension==1')) / len(
            df_src.query('diabetes==1')) * 100
    ]}, index=['Diabetes Negative', 'Diabetes Positive'])
st.table(percentage_stats_table)

st.divider()
st.write('Created by Charan Williams August, 2023')
st.caption(
    'Disclaimer: This application is for educational and demonstration purposes only and is not intended to provide actual '
    'medical advice. Always consult a licensed healthcare provider if you have any health concerns.')
