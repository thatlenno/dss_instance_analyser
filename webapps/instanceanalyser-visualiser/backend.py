from dataiku.customwebapp import *

# Access the parameters that end-users filled in using webapp config
# For example, for a parameter called "input_dataset"
# input_dataset = get_webapp_config()["input_dataset"]

import plotly.express as px
import dataiku
from dash import dcc, html
import pandas as pd

# This loads dummy data into a dataframe
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
}) \
 \
# Uncomment the following to read your own dataset
#dataset = dataiku.Dataset("YOUR_DATASET_NAME_HERE")
#df = dataset.get_dataframe()

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
