#open terminal (ctrl+j)
#cd house_price_prediction
#streamlit run app.py

import streamlit as st
import numpy as np
from joblib import load

def load_model(model_file): 
    model = load(model_file)
    return model


def predict(model, Bedrooms=0, Bathrooom=0, squarefeet=0,):
    x = np.array([[Bedrooms, Bathrooom, squarefeet]])
    y = model.predict(x)
    return y[0]

st.title("House Price Prediction")

with st.form(key = 'my_form'):
    bedrooms = st.number_input(label='Beds', value = 2, min_value = 1)
    bath = st.number_input(label='Baths', value = 2, min_value = 1)
    sqrft = st.number_input(label='Square Feet', value = 1000, min_value = 100, step = 100)
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    model = load_model('house_price.joblib')
    price = predict(model, bedrooms, bath, sqrft)
    st.success(f'Predicted Price: {int(price):,}')