import streamlit as st
from Explore_Page import show_analysis
from Predict_Page import show_prediction

page = st.sidebar("Analysis Or Prediction", ("Prediction", "Analysis"))

if page == "Prediction":
    show_prediction()
else:
    show_analysis()

