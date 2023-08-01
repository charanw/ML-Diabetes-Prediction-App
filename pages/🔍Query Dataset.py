import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Query Dataset - WGU Capstone",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state='auto'
)

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv", low_memory=False)
    return df

# Save the data as a state variable so that it can be accessed by other pages
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df_src = st.session_state.df

st.title("🔍 Search Dataset")
st.divider()
st.subheader('Show records where')
field = st.selectbox(label='Field',
                     options=sorted(list(df_src)), label_visibility='hidden', on_change=None)
radio_options_1 = [ 'Less than', 'Greater than', 'Equal to']
radio_options_2 = [ 'Equal to']
st.subheader('Is')
container = st.container()

value = st.selectbox(label='Value', label_visibility='hidden', options=sorted(df_src[field].unique().tolist()))

if type(value) == str:
    operator = container.radio(label='Operator', label_visibility='hidden', options=radio_options_2, horizontal=True)
else:
    operator = container.radio(label='Operator', label_visibility='hidden', options=radio_options_1, horizontal=True)
st.subheader('Number of records to show')
rows = st.number_input(label='Rows', label_visibility='hidden', min_value=1, value=10, max_value=len(df_src), step=10)
submitted = st.button('Go')

if submitted:
    st.divider()
    st.title('Results')
    st.subheader("Showing up to {rows} records where {field} is {operator} {value}:".format(rows=rows, field=field, operator=operator, value=value))
    if type(value) == str:
        st.dataframe(df_src.query('{field} == "{value}"'.format(field = str(field), value = value)).head(rows).sort_values(by=field))
        length = len(df_src.query('{field} == "{value}"'.format(field = str(field), value = value)))
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