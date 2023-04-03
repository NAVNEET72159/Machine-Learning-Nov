import streamlit as st
import numpy as np
from joblib import load, dump

def load_model(model_file):
    model = load(model_file)
    return model

def predict(model, temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi, region):
    x = np.array([[temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi, region]])
    y = model.predict(x)
    return y

st.title("Algerian Forest Fire Prediction")

with st.form(key='my_form'):
    temperature = st.number_input(
        label='Temperature', value=0, min_value=0
    )
    rh = st.number_input(
        label="RH", value=0, min_value=0
    )
    ws = st.number_input(
        label="WS", value=0, min_value=0
    )
    rain = st.number_input(
        label="rain", value=0, min_value=0
    )
    fmmc = st.number_input(
        label="FMMC", value=0, min_value=0
    )
    dmc = st.number_input(
        label="DMC", value=0, min_value=0
    )
    dc = st.number_input(
        label="DC", value=0, min_value=0
    )
    isi = st.number_input(
        label="ISI", value=0, min_value=0
    )
    bui = st.number_input(
        label="BUI", value=0, min_value=0
    )
    fwi = st.number_input(
        label="FWI", value=0, min_value=0
    )
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    model = load_model('Used_Device_Price_Prediction.joblib')
    value = predict(model, temperature, rh, ws, rain, fmmc, dmc, dc, isi, bui, fwi)
    st.success('Predicted Value: {y}')

    