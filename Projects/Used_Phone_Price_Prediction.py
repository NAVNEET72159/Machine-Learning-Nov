import streamlit as st
import numpy as np 
from joblib import load

brand_dict = load('brand_dict')

def load_model(model_file):
    model = load(model_file)
    return model

def predict(model, screen_size, _4g, _5g, rear_camera_mp, front_camera_mp, internal_memory, ram, battery, weight, days_used, normalized_new_price, device_brand):
    x = np.array([[screen_size, _4g, _5g, rear_camera_mp, front_camera_mp, internal_memory, ram, battery, weight, days_used, normalized_new_price, device_brand]])
    y = model.predict(x)
    return y[0]


st.title("Used Phone Price Prediction")

with st.form(key='my_form'):
    screen_size = st.number_input(
        label='Screen-Size', value=12.5, min_value=2.0)
    _4g = st.number_input("4G Supported (0 for No and 1 for Yes)", value=1, min_value=0)
    _5g = st.number_input(
        "5G Supported (0 for No and 1 for Yes)", value=1, min_value=0)
    rear_camera_mp = st.number_input(
        label='Rear Camera Pixel', value=120, min_value=2)
    front_camera_mp = st.number_input(
        label='Front Camera Pixel', value=32, min_value=2)
    internal_memory = st.number_input(
        label='Memory Size', value=512, min_value=16)
    ram = st.number_input(
        label='RAM', value=16, min_value=4)
    battery = st.number_input(
        label='Battery Capacity', value=1000, min_value=100)
    weight = st.number_input(
        label='Weight', value=1000, min_value=100)
    days_used_ = st.number_input(
        label='Total number of days used', value=1000, min_value=1)
    normalized_new_price = st.slider("Used Price", 0, 100000, 500)
    device_brand = st.selectbox("Device Brand", brand_dict)
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    model = load_model('Used_Device_Price_Prediction.joblib')
    price = predict(model, screen_size, _4g, _5g, rear_camera_mp, front_camera_mp, internal_memory, ram, battery, weight, days_used_, normalized_new_price, device_brand)
    st.success(f'Predicted Price: {int(price):,}')