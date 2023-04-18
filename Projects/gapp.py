import gradio as gr
from joblib import load
import numpy as np
import pandas as pd


def predict_price(
    sym, nl, make,
    fuel, aspiration, doors,
    body, wheels, engloc,
    wheelbase, length, width,
    height, curbweight, engtype,
    cyl, engsize, fuelsys,
    bore, stroke, compratio,
    hp, peakrpm, citympg,
    highwaympg
):
    # Load the model
    model = load('automobile_price_prediction.jb')

    # Create a dict array from the parameters
    data = {
        'symboling': [sym],
        'normalized-losses': [nl],
        'make': [make],
        'fuel-type': [fuel],
        'aspiration': [aspiration],
        'num-of-doors': [doors],
        'body-style': [body],
        'drive-wheels': [wheels],
        'engine-location': [engloc],
        'wheel-base': [wheelbase],
        'length': [length],
        'width': [width],
        'height': [height],
        'curb-weight': [curbweight],
        'engine-type': [engtype],
        'num-of-cylinders': [cyl],
        'engine-size': [engsize],
        'fuel-system': [fuelsys],
        'bore': [bore],
        'stroke': [stroke],
        'compression-ratio': [compratio],
        'horsepower': [hp],
        'peak-rpm': [peakrpm],
        'city-mpg': [citympg],
        'highway-mpg': [highwaympg],
    }
    Xinp = pd.DataFrame(data)
    print(Xinp)

    # Predict the price
    price = model.predict(Xinp)

    # return the price
    return price[0]

# Create the gradio interface


ui = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.inputs.Textbox(placeholder='Symboling', default=0,
                          numeric=True, label='Symboling'),
        gr.inputs.Textbox(placeholder='Normalized Losses',
                          default=100, numeric=True, label='Normalized Losses'),
        gr.inputs.Textbox(placeholder='Make', default='audi', label='make'),
        gr.inputs.Textbox(placeholder='Fuel Type',
                          default='gas', label='fuel'),
        gr.inputs.Textbox(placeholder='Aspiration',
                          default='std', label='aspiration'),
        gr.inputs.Textbox(placeholder='Number of Doors',
                          default='two', label='doors'),
        gr.inputs.Textbox(placeholder='Body Style',
                          default='sedan', label='body'),
        gr.inputs.Textbox(placeholder='Drive Wheels',
                          default='fwd', label='wheels'),
        gr.inputs.Textbox(placeholder='Engine Location',
                          default='front', label='engloc'),
        gr.inputs.Textbox(placeholder='Wheel Base', default=100,
                          numeric=True, label='wheelbase'),
        gr.inputs.Textbox(placeholder='Length', default=100,
                          numeric=True, label='length'),
        gr.inputs.Textbox(placeholder='Width', default=100,
                          numeric=True, label='width'),
        gr.inputs.Textbox(placeholder='Height', default=100,
                          numeric=True, label='height'),
        gr.inputs.Textbox(placeholder='Curb Weight',
                          default=100, numeric=True, label='curbweight'),
        gr.inputs.Textbox(placeholder='Engine Type',
                          default='dohc', label='engtype'),
        gr.inputs.Textbox(placeholder='Number of Cylinders',
                          default=4, numeric=True, label='cyl'),
        gr.inputs.Textbox(placeholder='Engine Size',
                          default=100, numeric=True, label='engsize'),
        gr.inputs.Textbox(placeholder='Fuel System',
                          default='mpfi', label='fuelsys'),
        gr.inputs.Textbox(placeholder='Bore', default=100,
                          numeric=True, label='bore'),
        gr.inputs.Textbox(placeholder='Stroke', default=100,
                          numeric=True, label='stroke'),
        gr.inputs.Textbox(placeholder='Compression Ratio',
                          default=100, numeric=True, label='compratio'),
        gr.inputs.Textbox(placeholder='Horsepower',
                          default=100, numeric=True, label='hp'),
        gr.inputs.Textbox(placeholder='Peak RPM', default=100,
                          numeric=True, label='peakrpm'),
        gr.inputs.Textbox(placeholder='City MPG', default=100,
                          numeric=True, label='citympg'),
        gr.inputs.Textbox(placeholder='Highway MPG',
                          default=100, numeric=True, label='highwaympg'),
    ],
    outputs="text",
)
ui.launch()
