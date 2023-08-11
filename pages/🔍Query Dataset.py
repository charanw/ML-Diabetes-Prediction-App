import pandas as pd
import streamlit as st

# Set configuration settings for the page
st.set_page_config(
    page_title="Query Dataset - WGU Capstone",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state='auto'
)


# Function to load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv", low_memory=False)
    return df


# Save the data as a state variable so that it can be accessed by other pages
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df_src = st.session_state.df

# Begin page content
st.title("🔍 Search Dataset")
st.divider()

# Input fields for querying the data set
st.subheader('Show records where')
field = st.selectbox(label='Field',
                     options=sorted(list(df_src)), label_visibility='hidden', on_change=None)
st.subheader('Is')

# Empty container to insert radio inputs into depending on data type of selected query value
container = st.container()

value = st.selectbox(label='Value', label_visibility='hidden', options=sorted(df_src[field].unique().tolist()))

# Options for the radio inputs to filter the data query
radio_options_1 = ['Less than', 'Greater than', 'Equal to']
radio_options_2 = ['Equal to']

if type(value) == str:
    operator = container.radio(label='Operator', label_visibility='hidden', options=radio_options_2, horizontal=True)
else:
    operator = container.radio(label='Operator', label_visibility='hidden', options=radio_options_1, horizontal=True)

# Input to determine how many records to show
st.subheader('Number of records to show')
rows = st.number_input(label='Rows', label_visibility='hidden', min_value=1, value=10, max_value=len(df_src), step=10)

# Button to submit the query
submitted = st.button('Go')

# Query the data on button click and display the results
if submitted:
    st.divider()
    st.title('Results')
    st.subheader("Showing up to {rows} records where {field} is {operator} {value}:".format(rows=rows, field=field,
                                                                                            operator=operator,
                                                                                            value=value))
    if type(value) == str:
        st.dataframe(
            df_src.query('{field} == "{value}"'.format(field=str(field), value=value)).head(rows).sort_values(by=field))
        length = len(df_src.query('{field} == "{value}"'.format(field=str(field), value=value)))
    else:
        if operator == 'Less than':
            st.dataframe(df_src.query('{field} < {value}'.format(field=str(field), value=value)).head(rows))
            length = len(df_src.query('{field} < {value}'.format(field=str(field), value=value)))
        elif operator == 'Greater than':
            st.dataframe(df_src.query('{field} > {value}'.format(field=str(field), value=value)).head(rows))
            length = len(df_src.query('{field} > {value}'.format(field=str(field), value=value)))
        elif operator == 'Equal to':
            st.dataframe(df_src.query('{field} == {value}'.format(field=str(field), value=value)).head(rows))
            length = len(df_src.query('{field} == {value}'.format(field=str(field), value=value)))
    st.subheader("There are {length} records total.".format(length=length))
st.divider()
st.write('Created by Charan Williams')
st.write('August 10th, 2023')
st.caption('Disclaimer: This application is for educational and demonstration purposes only and is not intended to provide actual '
         'medical advice. Always consult a licensed healthcare provider if you have any health concerns.')