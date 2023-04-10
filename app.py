import gradio as gr
from joblib import load
import numpy as np
import pandas as pd

def predict_price(gravity, ph, osmo, cond, urea, calc):
    model = load('Stone_Prediction.joblib')

    data = {
        'Gravity': [gravity],
        'PH': [ph],
        'OSMO': [osmo],
        'COND': [cond],
        'UREA': [urea],
        'CALC': [calc],
    }

    Xinp = pd.DataFrame(data)
    print(Xinp)

    stone = model.predict(Xinp)

    return stone[0]

ui = gr.Interface(
    fn=predict_price,inputs=[
        gr.inputs.Textbox(placeholder='gravity', default=0,
                          numeric=True, label='gravity (normal is between 1.002 - 1.030)'),
        gr.inputs.Textbox(placeholder='PH',
                          default=4.5, numeric=True, label='PH value (normal is between 4.5 - 8.0)'),
        gr.inputs.Textbox(placeholder='Osmolarity', default='500', numeric=True, label='Osmolarity (normal is between 500 mOsm/kg - 800 mOsm/kg)'),
        gr.inputs.Textbox(placeholder='Conductivity',
                          default='50', numeric=True, label='Conductivity (normal is between 50 - 1500 µS/cm or 0.05 - 1.5 mS/cm)'),
        gr.inputs.Textbox(placeholder='Urea',
                          default='12', numeric=True, label='Urea (normal is between 12 to 20 mg/dL or 3.6 to 7.1 mmol/L.)'),
        gr.inputs.Textbox(placeholder='Calc',
                          default='2.5', numeric=True, label='Calc (normal is between 100 to 300 mg/dL or 2.5 to 7.5 mmol/L)'),
    ],
    outputs="text",
)

if __name__ == "__main__":
    ui.launch(share=True)