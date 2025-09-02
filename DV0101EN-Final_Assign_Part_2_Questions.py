import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv'
)

# Initialize Dash app
app = dash.Dash(__name__)

# Dropdown options
dropdown_options = [
    {'label': 'Yearly Statistics', 'value': 'Yearly Statistics'},
    {'label': 'Recession Period Statistics', 'value': 'Recession Statistics'}
]

year_list = [i for i in range(1980, 2024)]

# App layout
app.layout = html.Div([
    html.H1("Automobile Sales Dashboard", style={'textAlign': 'center', 'color': '#503D36', 'font-size': 24}),
    
    html.Div([
        html.Label("Select Statistics:"),
        dcc.Dropdown(
            id='select-statistics',
            options=dropdown_options,
            value='Yearly Statistics'
        )
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Dropdown(
            id='select-year',
            options=[{'label': i, 'value': i} for i in year_list],
            value=2023
        )
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    html.Div(
        id='output-container',
        className='output-container',
        style={'marginTop': '20px'}
    )
])

# Enable/disable year dropdown based on statistics
@app.callback(
    Output('select-year', 'disabled'),
    Input('select-statistics', 'value')
)
def toggle_year_dropdown(stat_type):
    return False if stat_type == 'Yearly Statistics' else True

# Update graphs based on selection
@app.callback(
    Output('output-container', 'children'),
    [Input('select-statistics', 'value'), Input('select-year', 'value')]
)
def update_graphs(stat_type, year_selected):
    if stat_type == 'Recession Statistics':
        recession_data = data[data['Recession'] == 1]

        # Graph 1: Average Automobile Sales per Year during Recession
        yearly_rec = recession_data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        R_chart1 = dcc.Graph(figure=px.line(yearly_rec, x='Year', y='Automobile_Sales', title='Average Automobile Sales Over Recession Period'))

        # Graph 2: Average Vehicles Sold by Vehicle Type
        avg_vehicle_sales = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        R_chart2 = dcc.Graph(figure=px.bar(avg_vehicle_sales, x='Vehicle_Type', y='Automobile_Sales', title='Average Vehicles Sold by Vehicle Type During Recession'))

        # Graph 3: Advertising Expenditure Share by Vehicle Type
        exp_data = recession_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        R_chart3 = dcc.Graph(figure=px.pie(exp_data, names='Vehicle_Type', values='Advertising_Expenditure', title='Advertising Expenditure Share by Vehicle Type During Recession'))

        # Graph 4: Effect of Unemployment Rate on Vehicle Type and Sales
        unemp_data = recession_data.groupby(['unemployment_rate', 'Vehicle_Type'])['Automobile_Sales'].mean().reset_index()
        R_chart4 = dcc.Graph(
            figure=px.bar(
                unemp_data, 
                x='unemployment_rate', 
                y='Automobile_Sales', 
                color='Vehicle_Type',
                labels={'unemployment_rate':'Unemployment Rate','Automobile_Sales':'Average Automobile Sales'},
                title='Effect of Unemployment Rate on Vehicle Type and Sales'
            )
        )

        return [
            html.Div([R_chart1, R_chart2], style={'display': 'flex', 'justify-content': 'space-around'}),
            html.Div([R_chart3, R_chart4], style={'display': 'flex', 'justify-content': 'space-around'})
        ]

    elif stat_type == 'Yearly Statistics' and year_selected:
        yearly_data = data[data['Year'] == year_selected]

        # Graph 1: Yearly Automobile Sales
        yearly_avg = data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        Y_chart1 = dcc.Graph(figure=px.line(yearly_avg, x='Year', y='Automobile_Sales', title='Yearly Automobile Sales'))

        # Graph 2: Monthly Automobile Sales
        monthly_sales = yearly_data.groupby('Month')['Automobile_Sales'].sum().reset_index()
        Y_chart2 = dcc.Graph(figure=px.line(monthly_sales, x='Month', y='Automobile_Sales', title=f'Monthly Automobile Sales in {year_selected}'))

        # Graph 3: Average Vehicles Sold by Type
        avg_vehicle_sales = yearly_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        Y_chart3 = dcc.Graph(figure=px.bar(avg_vehicle_sales, x='Vehicle_Type', y='Automobile_Sales', title=f'Average Vehicles Sold by Vehicle Type in {year_selected}'))

        # Graph 4: Advertising Expenditure Pie Chart
        exp_yearly = yearly_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        Y_chart4 = dcc.Graph(figure=px.pie(exp_yearly, names='Vehicle_Type', values='Advertising_Expenditure', title=f'Total Advertising Expenditure by Vehicle Type in {year_selected}'))

        return [
            html.Div([Y_chart1, Y_chart2], style={'display':'flex', 'justify-content': 'space-around'}),
            html.Div([Y_chart3, Y_chart4], style={'display': 'flex', 'justify-content': 'space-around'})
        ]

    else:
        return None

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
