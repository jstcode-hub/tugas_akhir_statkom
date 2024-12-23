import streamlit as st

# load page
training_page = st.Page("./training.py", title="Training Page")
testing_page = st.Page("./testing.py", title="Testing Page")
form_predict = st.Page("./form_predict.py", title="Form Predict")

# Initialize the page
pg = st.navigation([training_page, testing_page, form_predict])
st.set_page_config(page_title="Tugas Akhir")

pg.run()