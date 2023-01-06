import streamlit as st
import numpy as np
from joblib import load


def load_model(model_file):
    model = load(model_file)
    return model


def predict(model, Bathrooms = 0, sqft_living = 0, grade = 0, sqft_above = 0, sqft_living15 = 0,
            sqft_basement = 0, view = 0, floors = 0, waterfront = 0, bedrooms = 0, yr_renovated = 0, lat = 0):
    x = np.array([[Bathrooms, sqft_living, grade, sqft_above, sqft_living15,
                   sqft_basement, view, floors, waterfront, bedrooms, yr_renovated, lat]])
    y = model.predict(x)
    return y[0]


st.title("KC House Price Prediction")

with st.form(key='my_form'):
    
    baths = st.number_input(label = 'Bathrooms', value = 2, min_value = 1)
    living_room_area = st.number_input(
        label='Area of living Room', value=13410, min_value=200, max_value=13410)
    grade = st.number_input(label = 'Grade', value = 3, min_value = 1)
    square_feet_above = st.number_input(
        label='Square Feet Above', value=9410, min_value=200, max_value=9410)
    square_feet_living15 = st.number_input(
        label='Square Feel Living 15', value=6210, min_value=400, max_value=6210)
    square_feet_basement = st.number_input(
        label='Area of basement', value=4820, min_value=0, max_value=4820)
    view = st.number_input(
        label='View', value=1, min_value=0, max_value=1)
    floor_number = st.number_input(
        label='Floor Number', value=4, min_value=0, max_value=4)
    waterfront_info = st.number_input(
        label='Waterfront', value=0, min_value=0, max_value=1)
    bedrooms = st.number_input(
        label='Number of Bedrooms', value=0, min_value=0, max_value=33)
    year_renovated = st.number_input(
        label='Year Renovated', value=0, min_value=0, max_value=2015)
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    model = load_model('kc_house_price.joblib')
    price = predict(model, baths, living_room_area, grade, square_feet_above,
                    square_feet_living15, square_feet_basement, view, floor_number, waterfront_info, bedrooms, year_renovated)
    st.success(f'Predicted Price: {int(price):,}')
