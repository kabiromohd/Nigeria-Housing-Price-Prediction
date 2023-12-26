#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle

import dash
from dash import Dash, Input, Output, State, dcc, html, callback

import xgboost as xgb
from xgboost import XGBRegressor
    
def population_density(df):
    
    population_level_1 = ['Anambra', 'Enugu', 'Imo', 'Lagos']
    population_level_2 = ['Abia', 'Kano', 'Rivers']
    population_level_3 = ['Akwa Ibom', 'Ebonyi', 'Ekiti', 'Osun']
    population_level_4 = ['Katsina', 'Ogun', 'Ondo']
    population_level_5 = ['Bauchi', 'Delta', 'Jigawa', 'Oyo']
    population_level_6 = ['Bayelsa', 'Edo', 'Gombe']
    population_level_7 = ['Cross River', 'Kaduna', 'Plateau', 'Sokoto']
    population_level_8 = ['Kebbi', 'Kogi', 'Zamfara']
    population_level_9 = ['Adamawa', 'Benue', 'Kwara', 'Nasarawa']
    population_level_10 = ['Borno', 'Niger', 'Taraba', 'Yobe']

    df['population_density_loc'] = df['loc'].apply(lambda x: "pop_level_10" if x in population_level_1
                                                   else "pop_level_9" if x in population_level_2
                                                   else "pop_level_8" if x in population_level_3
                                                   else "pop_level_7" if x in population_level_4
                                                   else "pop_level_6" if x in population_level_5
                                                   else "pop_level_5" if x in population_level_6
                                                   else "pop_level_4" if x in population_level_7
                                                   else "pop_level_3" if x in population_level_8
                                                   else "pop_level_2" if x in population_level_9
                                                   else "pop_level_1" if x in population_level_10
                                                   else "pop_level_0")
    return df
    
def region_loc(df):
    south_east = ['Anambra', 'Enugu', 'Imo', 'Abia', 'Ebonyi']
    south_west = ['Lagos', 'Ekiti', 'Osun', 'Ogun', 'Ondo', 'Oyo']
    south_south = ['Akwa Ibom', 'Rivers', 'Delta', 'Edo', 'Cross River', 'Bayelsa']
    north_west = ['Katsina', 'Kano', 'Jigawa', 'Kebbi', 'Zamfara', 'Niger', 'Kaduna', 'Sokoto']
    north_east = ['Bauchi', 'Gombe', 'Borno', 'Adamawa', 'Taraba', 'Yobe']
    north_central = ['Plateau', 'Kogi', 'Benue', 'Kwara', 'Nasarawa']

    df['region_loc'] = df['loc'].apply(lambda x: "south_east" if x in south_east
                                       else "south_west" if x in south_west
                                       else "south_south" if x in south_south
                                       else "north_west" if x in north_west
                                       else "north_east" if x in north_east
                                       else "north_central" if x in north_central
                                       else "0")
    return df
    


# ## WEB DEPLOYMENT WITH DASH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# Create server variable with Flask server object for use with gunicorn
server = app.server # Flask server

app.layout = html.Div(
    [
        html.H1("Housing Price Prediction in Nigeria"),
        html.H3("Select State of property"),
        dcc.Dropdown(
            options = ['Katsina', 'Ondo', 'Ekiti', 'Anambra', 'Kogi', 'Borno', 'Kwara', 'Osun', 'Kaduna', 'Ogun',
                       'Bayelsa', "Unknown_loc", 'Abia', 'Rivers', 'Taraba', 'Ebonyi', 'Kebbi', 'Enugu', 'Edo',
                       'Nasarawa', 'Delta', 'Kano', 'Yobe', 'Benue', 'Bauchi', 'Cross River', 'Niger', 'Adamawa',
                       'Plateau', 'Imo', 'Oyo', 'Zamfara', 'Sokoto', 'Jigawa', 'Gombe', 'Akwa Ibom', 'Lagos'],
            value = "Lagos",
            id = "demo-plots-dropdown"
        ),
        html.Div(id = "demo-plot-display"),
        html.H3("Select type of apartment"),
        dcc.Dropdown(
            options = ['Semi-detached duplex', 'Apartment', 'Detached duplex', 'Terrace duplex', 'Mansion',
                       'Bungalow', 'Penthouse', 'Townhouse', 'Flat', 'Cottage'],
            value = 'Semi-detached duplex',
            id = "demo-apartment-dropdown"            
        ),
        html.Div(id = "demo-apartment-display"),
        html.H1(" "),
        html.H3("Enter the other property Details:"),
        html.Div('Bedrooms:', style={'display': 'inline-block', 'margin-right': '200px'}),
        html.Div('Bathrooms:', style={'display': 'inline-block', 'margin-right': '150px'}),
        html.Div('Parking Space:', style={'display': 'inline-block'}),
        html.H3(" "),
        dcc.Input(id='para-a', type='number', value=2.0, step=1),
        dcc.Input(id='para-b', type='number', value=2.0, step=1),
        dcc.Input(id='para-c', type='number', value=1.0, step=1),
        html.Div(id='output-container'),
        html.H3(" "),
        html.H6("Note: Accuracy of predictions as compared to what is obtainable is real life is based on modelling data"),
        html.H3(" "),
        html.Div(id='output')
    ]
)

@app.callback(
    Output("demo-plot-display", "children"),
    Input("demo-plots-dropdown", "value")
)
def display_demo_state(state_name):     
    return state_name


@app.callback(
    Output("demo-apartment-display", "children"),
    Input("demo-apartment-dropdown", "value")
)
def display_demo_apartment_type(apartment_type): 
    return apartment_type


@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('para-a', 'value'),
    dash.dependencies.Input('para-b', 'value'),
    dash.dependencies.Input('para-c', 'value')]
)
def update_output(para_a, para_b, para_c):
    if None not in (para_a, para_b, para_c):
        try:
            return f'Other selected Parameters: {para_a}, {para_b}, {para_c}'
        except ValueError:
            return 'Invalid input. Please enter valid floats.'
    else:
        return 'Enter all parameters.'


@app.callback(
    Output('output', 'children'),
    [Input('demo-plot-display', 'children'),
     Input('demo-apartment-display', 'children'),
     Input('para-a', 'value'),
     Input('para-b', 'value'),
     Input('para-c', 'value')]
)
def predict_Price(state_name, apartment_type, para_a, para_b, para_c):
    input_file = "model_best.pkl"
    input_file2 = "dicv.pkl"
    input_file3 = "scale.pkl"

    with open(input_file, 'rb') as f_in: 
        model = pickle.load(f_in)

    with open(input_file2, 'rb') as f_in2: 
        dv = pickle.load(f_in2)
        
    with open(input_file3, 'rb') as f_in3: 
        scaler = pickle.load(f_in3)
        
    client = [{'loc':state_name,
               "title":apartment_type,
               "bedroom": para_a,
               "bathroom": para_b,
               "parking_space": para_c}
             ]
    
    client = pd.DataFrame(client)       
    client['luxury_indicator'] = client['bedroom'] / client['bathroom']
    client['size_indicator'] = client['bedroom'] + client['bathroom'] + client['parking_space']
    client['luxury_by_size_indicator'] = client['luxury_indicator'] * client['size_indicator']
    client = population_density(client)
    client = region_loc(client)
    
    train_dicts = client.to_dict(orient='records')
    X_test = dv.transform(train_dicts)
    
    X_test = scaler.transform(X_test)
    
    y_pred = model.predict(X_test)
    result = np.expm1(y_pred).round(2)
    return f'Housing Price Prediction for "{state_name}" is: N{result}'
    
    
if __name__ == '__main__':
    app.run_server(debug = True, host = '0.0.0.0', port = 8096)
