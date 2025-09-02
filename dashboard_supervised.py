import pandas as pd
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

df = pd.read_csv(r"E:\New folder\ML_Projects\Supervised_Models.csv")

cols = ["Global_active_power","Global_reactive_power","Voltage","Global_intensity",
        "Sub_metering_1","Sub_metering_2","Sub_metering_3"]
df[cols] = df[cols].apply(pd.to_numeric)

# Optional: sample 10% to speed up
df = df.sample(frac=0.1, random_state=42)

# Aggregate daily
# Convert DateTime column explicitly
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Drop rows where conversion failed
df = df.dropna(subset=['DateTime'])

# Now resample by day
daily_df = df.set_index('DateTime').resample('D').mean().dropna()


# -------------------------
# 2. Create targets
# -------------------------
daily_df["next_day_consumption"] = daily_df["Global_active_power"].shift(-1)
daily_df["energy_plan"] = pd.cut(daily_df["Global_active_power"], bins=[0,1,3,6], labels=["Plan A","Plan B","Plan C"])
daily_df = daily_df.dropna()

# -------------------------
# 3. Train models
# -------------------------
X = daily_df[["Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]]

# Regression
y_reg = daily_df["next_day_consumption"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_r, y_train_r)
y_pred_r = regressor.predict(X_test_r)

# Classification
y_clf = daily_df["energy_plan"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_c, y_train_c)
y_pred_c = classifier.predict(X_test_c)

# -------------------------
# 4. Dash app
# -------------------------
app = Dash(__name__)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab1', children=[
        dcc.Tab(label='Daily Consumption', value='tab1'),
        dcc.Tab(label='Regression Predictions', value='tab2'),
        dcc.Tab(label='Classification Results', value='tab3'),
        dcc.Tab(label='Feature Correlation', value='tab4'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content','children'), Input('tabs','value'))
def render_content(tab):
    if tab=='tab1':
        fig = px.line(daily_df, x=daily_df.index, y="Global_active_power", title="Daily Global Active Power")
        return html.Div([dcc.Graph(figure=fig)])
    elif tab=='tab2':
        fig = px.scatter(x=y_test_r, y=y_pred_r, labels={'x':'Actual','y':'Predicted'}, title="Regression: Actual vs Predicted")
        return html.Div([dcc.Graph(figure=fig)])
    elif tab=='tab3':
        fig = px.histogram(y_pred_c, title="Predicted Energy Plan Distribution")
        return html.Div([dcc.Graph(figure=fig)])
    elif tab=='tab4':
        corr = daily_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
        return html.Div([dcc.Graph(figure=fig)])

if __name__=="__main__":
    app.run(debug=True)
