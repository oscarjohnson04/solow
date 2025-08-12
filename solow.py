import streamlit as st
import wbdata
import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LIS = st.number_input("Labour Income Share (Will help determine capital per capita):", min_value=0, max_value=1, value=0.5, step=0.01)
S = st.number_input("Savings Rate:", min_value=0, max_value=1, value=0.2, step=0.01)
D = st.number_input("Depreciation Rate:", min_value=0, max_value=1, value=0.05, step=0.01)

# Set time period
start_date = dt.datetime(1970, 1, 1)
end_date = dt.datetime(2024, 1, 1)

# Indicators:
indicators = {
    "SL.TLF.TOTL.IN": "Labour_Force",   # Labour force (total)
    "NY.GDP.MKTP.KD": "Real_GDP"        # GDP constant 2015 US$
}

# Fetch data for all countries
data = wbdata.get_dataframe(indicators, date=(start_date, end_date))

# Reset index (multi-index -> columns)
data = data.reset_index()

exclude_list = ["Africa Eastern and Southern", "Africa Western and Central", "Arab World",
                "Caribbean small states", "Central Europe and the Baltics", "Early-demographic dividend",
                "East Asia & Pacific", "East Asia & Pacific (IDA & IBRD countries)", "East Asia & Pacific (excluding high income)",
                "Euro area", "Europe & Central Asia", "Europe & Central Asia (IDA & IBRD countries)", "Europe & Central Asia (excluding high income)",
                "Fragile and conflict affected situations", "Heavily indebted poor countries (HIPC)", "High income",
                "IBRD only", "IDA & IBRD total", "IDA blend", "IDA only", "IDA total",
                "Late-demographic dividend", "Latin America & Caribbean", "Latin America & Caribbean (excluding high income)",
                "Latin America & the Caribbean (IDA & IBRD countries)", "Least developed countries: UN classification",
                "Low & middle income", "Low income", "Lower middle income", "Middle East, North Africa, Afghanistan & Pakistan"
                "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)", "Middle East, North Africa, Afghanistan & Pakistan (excluding high income)",
                "Middle income", "North America", "Not classified", "OECD members", "Other small states", "Pacific island small states",
                "Post-demographic dividend", "Pre-demographic dividend", "Small states", "South Asia", "South Asia (IDA & IBRD)",
                "Sub-Saharan Africa", "Sub-Saharan Africa (IDA & IBRD countries)", "Sub-Saharan Africa (excluding high income)",
                "Upper middle income", "Middle East, North Africa, Afghanistan & Pakistan", "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)"]
                
                              
data = data[~data['country'].isin(exclude_list)]

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], format='%Y')

# Sort by country and date
data = data.sort_values(by=["country", "date"], ascending=[True, True])

data["Labour_Force_Growth"] = (
    data.groupby("country")["Labour_Force"]
    .pct_change() 
)

data["Mean_Labour_Growth"] = (
    data.groupby("country")["Labour_Force_Growth"].transform("mean")
)

data["GDPi"] = data["Real_GDP"] / data["Labour_Force"]

df = data[['country', 'date', 'Labour_Force', 'Real_GDP', 'Mean_Labour_Growth', 'GDPi']]
solow_df = df.loc[df.groupby("country")["date"].idxmax()].reset_index(drop=True)

# Preview
st.write(solow_df.tail(2050))
