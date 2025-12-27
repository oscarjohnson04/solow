import streamlit as st
import wbdata
import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objects as go

# --- Page config must be first Streamlit call ---
st.set_page_config(page_title="Solow-Romer Growth Model Explorer", layout="wide")
st.title("Solow-Romer Growth Model Explorer")

# -----------------------
# Parameters & Indicators
# -----------------------
START_DATE = dt.datetime(1970, 1, 1)
END_DATE = dt.datetime(2024, 1, 1)
INDICATORS = {
    "SP.POP.TOTL": "Population",   # Labour force (total)
    "NY.GDP.MKTP.KD": "Real_GDP",        # GDP (constant 2015 US$)
    "NY.GDS.TOTL.ZS": "Savings_Rate"
}

# -----------------------
# Data loading & cleaning
# -----------------------
@st.cache_data 
def load_and_process_data(start_date, end_date, indicators):
    # Fetch data from World Bank API
    data = wbdata.get_dataframe(indicators, date=(start_date, end_date))
    data = data.reset_index()

    # List of regions/aggregates to exclude
    exclude_list = [
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
        "Upper middle income"
    ]

    # Filter out non-country aggregates
    data = data[~data['country'].isin(exclude_list)]

    # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y')

    # Sort by country and date
    data = data.sort_values(by=['country', 'date'])

    # Calculate labour force growth rate (percent change)
    data['Population_Growth'] = data.groupby('country')['Population'].pct_change()

    # Calculate mean labour force growth for each country (same for all rows of that country)
    def weighted_pop_growth(group, span=10):
        return group['Population_Growth'].ewm(span=span, adjust=False).mean().iloc[-1]

    mean_growth = (
        data.groupby('country')
        .apply(weighted_pop_growth, span=10)
        .rename('Mean_Population_Growth')
    )

    data = data.merge(mean_growth, on='country')

    # GDP per worker (rounded to 2 decimals)
    data['GDPi'] = (data['Real_GDP'] / data['Population']).round(2)

    data['S_Rate'] = (data['Savings_Rate'] / 100).round(2)

    # For each country, keep only the most recent data point
    latest = data.loc[data.groupby('country')['date'].idxmax()].reset_index(drop=True)

    return latest

solow_df = load_and_process_data(START_DATE, END_DATE, INDICATORS)

# -----------------------
# Sidebar / Inputs
# -----------------------
with st.sidebar:
    st.subheader("Model parameters")
    display_mode = st.radio("Display Units:", ["Index (Base 100)", "Absolute ($ Value)"])    
    ky_ratio = st.slider("Initial Capital-Output Ratio (k/y)", 1.0, 6.0, 3.0, 
                         help="World average is ~3.0. This determines how much capital exists relative to GDP.")
    
    alpha = st.number_input("Capital share (α)", 0.01, 0.99, 0.33)
    
    st.markdown("---")
    st.write("**Romer R&D Settings**")
    lambda_RD = st.number_input("R&D effectiveness (λ)", min_value=0.0, max_value=1.0, value=0.05, step=0.10, format="%.2f")
    phi = st.number_input("R&D returns to scale (φ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    theta = st.number_input("R&D labor share (θ)", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    delta = st.number_input("Capital Depreciation (δ)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
    T = st.number_input("Simulation periods (T)", min_value=1, max_value=2000, value=100, step=1)

# Country selection (top of main)
default_country = "Canada"

try:
    default_ix = list(countries).index(default_country)
except ValueError:
    default_ix = 0
countries = solow_df["country"].sort_values().unique()
selected_country = st.selectbox("Select a country:", countries, index=default_ix)

# Current country row
row = solow_df.loc[solow_df["country"] == selected_country].iloc[0]
y_data = float(row["GDPi"])                   # observed GDP per worker (latest)
n = float(row["Mean_Population_Growth"]) if pd.notna(row["Mean_Population_Growth"]) else 0.01
N0 = float(row["Population"])
s = float(row["S_Rate"]) if pd.notna(row["S_Rate"]) else 0.20
country_name = row["country"]
latest_year = row["date"].year if not pd.isna(row["date"]) else "N/A"

# Guardrails / validation
if alpha <= 0 or alpha >= 1:
    st.error("Capital share α must be between 0 and 1.")
    st.stop()
if y_data <= 0 or N0 <= 0:
    st.error("Selected country has invalid latest values (GDP per worker or labour force).")
    st.stop()

# -----------------------
# Solow helpers
# -----------------------
A0_calibrated = y_data / ((ky_ratio * y_data) ** alpha)

k0 = ky_ratio * y_data

def romer_A_path(A0_calibrated, lambda_RD, phi, theta, N_path):
    """
    Romer-style endogenous TFP growth with balanced growth.
    """
    T = len(N_path)    
    A = np.zeros(T)    
    A[0] = A0_calibrated   
    N_norm = N_path / N_path[0]        
    for t in range(1, T):
        L_A = theta * N_norm[t-1]
        gA = lambda_RD * (L_A ** phi)
        A[t] = A[t-1] * (1 + gA)
    
    return A

def solow_k_path(k0, A_path, alpha, s, delta, n, T):
    """k_{t+1} = [ s *A_path *k_t^α + (1-δ)k_t ] / (1+n)"""
    k = np.empty(T, dtype=float)
    k[0] = k0
    for t in range(1, T):
        k[t] = (s * A_path[t-1] * k[t-1] ** alpha) / (1.0 + n) + ((1.0 - delta) * k[t-1]) / (1.0 + n)
    return k

def lf_path(N0, n, T):
    """Labour force path with constant growth n."""
    return N0 * (1.0 + n) ** np.arange(T)

# -----------------------
# Build paths
# -----------------------
N_path = lf_path(N0, n, T)
A_path = romer_A_path(A0_calibrated, lambda_RD, phi, theta, N_path)
k_path = solow_k_path(k0, A_path, alpha, s, delta, n, T)

# Vectorized macro identities
y_path = A_path * (k_path ** alpha)          # output per (effective) worker
i_path = s * y_path                     # investment per worker
c_path = (1.0 - s) * y_path             # consumption per worker
GDP_path = y_path * N_path              # total output

def to_index(x, base=0):
    return 100.0 * x / x[base]

if display_mode == "Index (Base 100)":
    k_plot = to_index(k_path)
    y_plot = to_index(y_path)
    i_plot = to_index(i_path)
    c_plot = to_index(c_path)
    A_plot = to_index(A_path)
    GDP_plot = to_index(GDP_path)
    y_label = "Index (Start = 100)"
else:
    k_plot = k_path
    y_plot = y_path
    i_plot = i_path
    c_plot = c_path
    A_plot = A_path
    GDP_plot = GDP_path
    y_label = "Value (USD / Units)"

# -----------------------
# Headline & country panel
# -----------------------
st.markdown(f"**Latest data for {country_name} (year: {latest_year})**")
summary_df = pd.DataFrame({
    "Variable": [
        "Latest year",
        "Population",
        "Real GDP",
        "GDP per capita",
        "Mean population growth (n)",
        "Savings Rate"
    ],
    "Value": [
        latest_year,
        N0,
        row["Real_GDP"],
        y_data,
        n,
        s
    ]
})

def format_value(val, variable_name):
    if variable_name in ["Mean population growth (n)", "Savings Rate"]:
        return f"{val*100:.2f}%"  # percentage
    elif variable_name == "Real GDP" or variable_name == "GDP per capita":
        return f"${val:,.2f}"      # commas + 2 decimals
    elif variable_name == "Population":
        return f"{int(val):,}" 
    elif variable_name == "Latest year":
        return f"{int(val):}"
    else:
        return val    

summary_df["Value"] = [format_value(v, vn) for v, vn in zip(summary_df["Value"], summary_df["Variable"])]

st.dataframe(summary_df)

time = np.arange(T)

def make_line(y, title, ylab):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode="lines", name=title))
    fig.update_layout(title=title, xaxis_title="Periods", yaxis_title=ylab, template="plotly_white")
    return fig

col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(make_line(k_plot, f"Capital per capita (k) (Index = 100) — {country_name}", "k"), use_container_width=True)
    st.plotly_chart(make_line(y_plot, f"Output per capita (y) (Index = 100) — {country_name}", "y"), use_container_width=True)
    st.plotly_chart(make_line(GDP_plot, f"Total GDP (Index = 100) — {country_name}", "GDP"), use_container_width=True)

with col_right:
    st.plotly_chart(make_line(i_plot, f"Investment per capita (i) (Index = 100) — {country_name}", "i"), use_container_width=True)
    st.plotly_chart(make_line(c_plot, f"Consumption per capita (c) (Index = 100) — {country_name}", "c"), use_container_width=True)
    st.plotly_chart(make_line(A_plot, f"R&D Growth (Index = 100) — {country_name}", "A"), use_container_width=True)
# -----------------------
# Download results
# ----------------

paths_df = pd.DataFrame({
    "t": time,
    "k": k_plot,
    "y": y_plot,
    "i": i_plot,
    "c": c_plot,
    "N": N_path,
    "GDP": GDP_plot,
    "TFP": A_plot
})
st.download_button(
    "Download simulated paths (CSV)",
    data=paths_df.to_csv(index=False).encode("utf-8"),
    file_name=f"solow_paths_{country_name}.csv",
    mime="text/csv"
)

with st.expander("ℹ️ Information about the model"):
    st.write("The model is a Solow growth model that includes endogenous TFP growth based on Paul Romer's research.  \n"
            "Calculations for the model can be found in the GitHub repo.  \n"
            "For population growth, more weight is put on more recent data.")
