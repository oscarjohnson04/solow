import streamlit as st
import wbdata
import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objects as go

st.title("Solow Growth Model Explorer")

# Data loading parameters
start_date = dt.datetime(1970, 1, 1)
end_date = dt.datetime(2024, 1, 1)
indicators = {
    "SL.TLF.TOTL.IN": "Labour_Force",
    "NY.GDP.MKTP.KD": "Real_GDP"
}

@st.cache_data(show_spinner=True)
def load_and_process_data():
    # Fetch data
    data = wbdata.get_dataframe(indicators, date=(start_date, end_date))
    data = data.reset_index()

    # ICt of regions/aggregates to exclude
    exclude_ICt = [
        "Africa Eastern and Southern", "Africa Western and Central", "Arab World",
        "Caribbean small states", "Central Europe and the Baltics", "Early-demographic dividend",
        "East Asia & Pacific", "East Asia & Pacific (IDA & IBRD countries)", "East Asia & Pacific (excluding high income)",
        "Euro area", "Europe & Central Asia", "Europe & Central Asia (IDA & IBRD countries)", "Europe & Central Asia (excluding high income)",
        "Fragile and conflict affected situations", "Heavily indebted poor countries (HIPC)", "High income",
        "IBRD only", "IDA & IBRD total", "IDA blend", "IDA only", "IDA total",
        "Late-demographic dividend", "Latin America & Caribbean", "Latin America & Caribbean (excluding high income)",
        "Latin America & the Caribbean (IDA & IBRD countries)", "Least developed countries: UN classification",
        "Low & middle income", "Low income", "Lower middle income", "Middle East, North Africa, Afghanistan & Pakistan",
        "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)", "Middle East, North Africa, Afghanistan & Pakistan (excluding high income)",
        "Middle income", "North America", "Not classified", "OECD members", "Other small states", "Pacific island small states",
        "Post-demographic dividend", "Pre-demographic dividend", "Small states", "South Asia", "South Asia (IDA & IBRD)",
        "Sub-Saharan Africa", "Sub-Saharan Africa (IDA & IBRD countries)", "Sub-Saharan Africa (excluding high income)",
        "Upper middle income", "Middle East, North Africa, Afghanistan & Pakistan", "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)"
    ]

    # Filter out non-countries
    data = data[~data['country'].isin(exclude_ICt)]

    # Convert 'date' to datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y')

    # Sort by country and date
    data = data.sort_values(by=['country', 'date'])

    # Calculate labour force growth rate (pct_change)
    data['Labour_Force_Growth'] = data.groupby('country')['Labour_Force'].pct_change()

    # Mean labour force growth (same value for each row in a country)
    data['Mean_Labour_Growth'] = data.groupby('country')['Labour_Force_Growth'].transform('mean')

    # GDP per worker rounded to 2 decimals
    data['GDPi'] = (data['Real_GDP'] / data['Labour_Force']).round(2)

    # For each country, keep only most recent data point
    latest = data.loc[data.groupby('country')['date'].idxmax()].reset_index(drop=True)

    return latest

solow_df = load_and_process_data()

# Country selection
countries = solow_df['country'].sort_values().unique()
selected_country = st.selectbox("Select a country:", countries)

IC = st.number_input(
    "Importance of Capital:",
    min_value=0.0, max_value=1.0, value=0.5, step=0.001,
    format="%.2f"
)
A = st.number_input("Level of Technology:", min_value=1.0, max_value=100.0, value=10.0, step=0.1)
S = st.number_input("Savings Rate:", min_value=0.0, max_value=1.0, value=0.2, step=0.001, format="%.2f")
D = st.number_input("Depreciation Rate:", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.2f")
T = st.number_input("Periods:", min_value=1, max_value=1000, value=100, step=1)

country_data = solow_df[solow_df['country'] == selected_country].iloc[0]

# Extract parameters
n = country_data['Mean_Labour_Growth'] if not pd.isna(country_data['Mean_Labour_Growth']) else 0.01
Ki = country_data['GDPi']
country_name = country_data['country']

# Show selected country summary
st.markdown(f"### Data for {country_name}")
st.write(country_data[['date', 'Labour_Force', 'Real_GDP', 'Mean_Labour_Growth', 'GDPi']])

# Calculate Ki rounded with user IC
Ki = round((Ki ** (1 / IC))/A, 4)

def solow_model(A, n, D, Ki, S, IC, T):
    k = np.zeros(T)
    k[0] = Ki
    for t in range(1, T):
        k[t] = (S / (1 + n)) * (A*k[t-1] ** IC) + k[t-1] * (1 - D) / (1 + n)
    return k

# Run model
k_path = solow_model(A, n, D, Ki, S, IC, T)

def ysolow_model(k_path, A, IC):
    y = np.zeros(T)
    y[0] = A*Ki**IC
    for t in range(1, T):
        y[t] = A * k_path[t-1]**IC
    return y
    
y_path = ysolow_model(k_path, A, IC)

def isolow_model(k_path, S, A, IC):
    I = np.zeros(T)
    I[0] = S*A*Ki**IC
    for t in range(1, T):
        I[t] = S*A*k_path[t-1]**IC
    return I

i_path = isolow_model(k_path, S, A, IC)

def csolow_model(k_path, S, A, IC):
    C = np.zeros(T)
    C[0] = (1-S)*A*Ki**IC
    for t in range(1, T):
        C[t] = (1-S)*A*k_path[t-1]**IC
    return C

c_path = csolow_model(k_path, S, A, IC)

def wsolow_model(k_path, A, IC):
    W = np.zeros(T)
    W[0] = (1-IC)*A*Ki**IC
    for t in range(1, T):
        W[t] = (1-IC)*A*k_path[t-1]**IC
    return W

w_path = wsolow_model(k_path, A, IC)

# Plot
time = list(range(T))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time,
    y=k_path,
    mode='lines',
    name='Capital per effective worker',
    line=dict(color='blue')
))
fig.update_layout(
    title=f"Solow Capital Growth Model - {country_name}",
    xaxis_title='Periods',
    yaxis_title='Capital per effective worker (k)',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=time,
    y=y_path,
    mode='lines',
    name='Output per effective worker',
    line=dict(color='blue')
))
fig1.update_layout(
    title=f"Solow Output Growth Model - {country_name}",
    xaxis_title='Periods',
    yaxis_title='Output per effective worker (Y)',
    template='plotly_white'
)

st.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=time,
    y=i_path,
    mode='lines',
    name='Investment per effective worker',
    line=dict(color='blue')
))
fig2.update_layout(
    title=f"Solow Investment Growth Model - {country_name}",
    xaxis_title='Periods',
    yaxis_title='Investment per effective worker (I)',
    template='plotly_white'
)

st.plotly_chart(fig2, use_container_width=True)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=time,
    y=c_path,
    mode='lines',
    name='Consumption per effective worker',
    line=dict(color='blue')
))
fig3.update_layout(
    title=f"Solow Consumption Growth Model - {country_name}",
    xaxis_title='Periods',
    yaxis_title='Consumption per effective worker (C)',
    template='plotly_white'
)

st.plotly_chart(fig3, use_container_width=True)

fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=time,
    y=w_path,
    mode='lines',
    name='Wage per effective worker',
    line=dict(color='blue')
))
fig4.update_layout(
    title=f"Solow Wage Growth Model - {country_name}",
    xaxis_title='Periods',
    yaxis_title='Wage per effective worker (W)',
    template='plotly_white'
)

st.plotly_chart(fig4, use_container_width=True)

