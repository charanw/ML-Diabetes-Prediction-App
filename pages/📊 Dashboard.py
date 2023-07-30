import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from plotly.subplots import make_subplots

import streamlit as st

st.set_page_config(
    page_title="Dashboard - WGU Capstone",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state='auto'
)

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv", low_memory=False)
    return df

@st.cache_data
def initialize():
    confusion_matrix = joblib.load("diabetes_confusion_matrix.save")
    return confusion_matrix

confusion_matrix= initialize()

# Create accuracy table

false_negative = confusion_matrix[1][0]
true_negative = confusion_matrix[0][0]
true_positive = confusion_matrix[1][1]
false_positive = confusion_matrix[0][1]

confusion_table = go.Figure(
    data = [go.Bar(
        x = ["False Negative", "True Negative" , "True Positive", "False Positive"],
        y = [false_negative, true_negative, true_positive, false_positive])], )

# Save the data as a state variable so that it can be accessed by other pages
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df_src = st.session_state.df

st.title("📊 Data Visualization")
st.divider()

st.header("Model Performance")
st.divider()
scores = joblib.load("training_and_testing_scores.save")
training_score = scores[0]
testing_score = scores[1]
st.subheader("Training Set Score = {}%".format(round(training_score, 3)*100))
st.subheader("Testing Set Score = {}%".format(round(testing_score, 3)*100))

st.plotly_chart(confusion_table, use_container_width=True, height=500)

# Create 2 column layout
col1, col2 = st.columns(2, gap='large')

# Create gender breakdown dataframe and pie chart
diabetes_negative_gender_totals = df_src.query('diabetes == 0')['gender'].value_counts().reset_index()
col1.plotly_chart(px.pie(data_frame=diabetes_negative_gender_totals, names="index", values='gender', title='Diabetes Negative Gender Breakdown', color='index', color_discrete_map={'Male': '#57799E', 'Female': '#DAA49A', 'Other': '#41818B'}, hover_name='index', hover_data='gender', width=750, height=750), use_container_width=True, height=500)


diabetes_positive_gender_totals = df_src.query('diabetes == 1')['gender'].value_counts().reset_index()
col2.plotly_chart(px.pie(data_frame=diabetes_positive_gender_totals, names="index", values='gender', title='Diabetes Positive Gender Breakdown', color='index', color_discrete_map={'Male': '#57799E', 'Female': '#DAA49A', 'Other': '#41818B'}, hover_name='index', hover_data='gender', width=750, height=750), use_container_width=True, height=500)

# Create dataframe with all bmi data and only bmi data of those who have diabetes
diabetes_negative_bmi_data = go.Histogram(x=df_src.query('diabetes == 0')['bmi'], marker=dict(color='#6E8EAF'), nbinsx=20, name="Diabetes Negative")
diabetes_positive_bmi_data = go.Histogram(x=df_src.query('diabetes == 1')['bmi'], marker=dict(color='#F52D00', opacity= .4), nbinsx=20, name="Diabetes Positive")

# Create histogram to plot bmi distribution
bmi_hist = make_subplots(specs=[[{"secondary_y": True}]])
bmi_hist.add_trace(diabetes_negative_bmi_data, secondary_y=False)
bmi_hist.add_trace(diabetes_positive_bmi_data, secondary_y=True)
bmi_hist.update_layout(title='BMI Distribution', xaxis_title='BMI', yaxis_title='Count', legend_title='Key', width=1200, height=700)
col1.plotly_chart(bmi_hist, use_container_width=True, height=300)

# Create dataframe with all A1C data and only A1C data of those who have diabetes
diabetes_negative_a1c_data = go.Histogram(x=df_src.query('diabetes == 0')['HbA1c_level'], marker=dict(color='#6E8EAF'), nbinsx=10, name="Diabetes Negative")
diabetes_positive_a1c_data= go.Histogram(x=df_src.query('diabetes == 1')['HbA1c_level'], marker=dict(color='#F52D00', opacity= .4), nbinsx=10, name="Diabetes Positive")

# Create histogram to plot A1C distribution
a1c_hist = make_subplots(specs=[[{"secondary_y": True}]])
a1c_hist.add_trace(diabetes_negative_a1c_data, secondary_y=False)
a1c_hist.add_trace(diabetes_positive_a1c_data, secondary_y=True)
a1c_hist.update_layout(title='A1C Distribution', xaxis_title='A1C', yaxis_title='Count', legend_title='Key', width=1200, height=700)


col2.plotly_chart(a1c_hist, use_container_width=True, height=300)

# Create dataframe with all age data and only age data of those who have diabetes
diabetes_negative_age_data = go.Histogram(x=df_src.query('diabetes == 0')['age'], marker=dict(color='#6E8EAF'), nbinsx=100, name="Diabetes Negative")
diabetes_positive_age_data= go.Histogram(x=df_src.query('diabetes == 1')['age'], marker=dict(color='#F52D00', opacity= .4), nbinsx=100, name="Diabetes Positive")

# Create histogram to plot A1C distribution
age_hist = make_subplots(specs=[[{"secondary_y": True}]])
age_hist.add_trace(diabetes_negative_age_data, secondary_y=False)
age_hist.add_trace(diabetes_positive_age_data, secondary_y=True)
age_hist.update_layout(title='Age Distribution', xaxis_title='Age', yaxis_title='Count', legend_title='Key', width=1200, height=700)
st.plotly_chart(age_hist, use_container_width=True, height=500)

"Let's see what the first few rows look like:"
st.write(df_src.head(5))
"Let's see what's filled out:"
(df_src.count()/len(df_src))*100
"Let's get more information. Let's see the counts of each value:"
for i in range(0, len(df_src.columns)):
    st.write(df_src.columns[i])
    st.write(df_src.iloc[:,i].value_counts())

"Lets see what percentage of those who have diabetes are male:"

st.write(len(df_src.query("diabetes == 1 and gender=='Male'"))/len(df_src.query('diabetes==1'))*100)

"Lets see what percentage of those who have diabetes are female:"

st.write(len(df_src.query("diabetes == 1 and gender == 'Female'"))/len(df_src.query('diabetes==1'))*100)

"Lets see the average age for those who don't have diabetes:"

st.write(df_src.query('diabetes == 0')['age'].mean())

"Lets see the average age for those who do have diabetes:"

st.write(df_src.query('diabetes == 1')['age'].mean())

"Lets see the average bmi for those who don't have diabetes:"

st.write(df_src.query('diabetes == 0')['bmi'].mean())

"Lets see the average bmi for those who do have diabetes:"

st.write(df_src.query('diabetes == 1')['bmi'].mean())

"Lets see the average A1C for those who don't have diabetes:"

st.write(df_src.query('diabetes== 0')['HbA1c_level'].mean())

"Lets see the average a1c for those who do have diabetes:"

st.write(df_src.query('diabetes == 1')['HbA1c_level'].mean())

"Lets see the average blood glucose level for those who don't have diabetes:"

st.write(df_src.query('diabetes== 0')['blood_glucose_level'].mean())

"Lets see the average blood glucose level for those who do have diabetes:"

st.write(df_src.query('diabetes == 1')['blood_glucose_level'].mean())

"Lets see what percentage of those who have diabetes have heart disease:"

st.write(len(df_src.query('diabetes == 1 and heart_disease==1'))/len(df_src.query('diabetes==1'))*100)

"Lets see what percentage of those who have diabetes have hypertension:"

st.write(len(df_src.query('diabetes == 1 and hypertension==1'))/len(df_src.query('diabetes==1'))*100)