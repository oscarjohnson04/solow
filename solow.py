import streamlit as st
import wbdata
import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objects as go

# --- Page config must be first Streamlit call ---
st.set_page_config(page_title="Solow Growth Model Explorer", layout="wide")
st.title("Solow Growth Model Explorer")

# -----------------------
# Parameters & Indicators
# -----------------------
START_DATE = dt.datetime(1970, 1, 1)
END_DATE = dt.datetime(2024, 1, 1)
INDICATORS = {
    "SL.TLF.TOTL.IN": "Labour_Force",   # Labour force (total)
    "NY.GDP.MKTP.KD": "Real_GDP"        # GDP (constant 2015 US$)
}

# -----------------------
# Data loading & cleaning
# -----------------------
@st.cache_data(show_spinner=True)
def load_and_process_data(start_date, end_date, indicators):
    """
    Download World Bank data, keep real countries only, compute growth & GDP per worker,
    and return the most recent observation per country.
    """
    # Fetch all entities
    raw = wbdata.get_dataframe(indicators, date=(start_date, end_date)).reset_index()

    # Keep only real countries (exclude aggregates)
    countries_meta = wbdata.get_country()  # list of dicts
    real_country_names = {c["name"] for c in countries_meta if c.get("region", {}).get("id") != "NA"}
    data = raw[raw["country"].isin(real_country_names)].copy()

    # Ensure proper dtypes
    data["date"] = pd.to_datetime(data["date"], format="%Y", errors="coerce")
    data = data.sort_values(["country", "date"])

    # Drop rows missing core inputs before calculations
    data = data.dropna(subset=["Labour_Force", "Real_GDP"])

    # Labour force growth (per year, as a rate)
    data["Labour_Force_Growth"] = (
        data.groupby("country")["Labour_Force"].pct_change()
    )

    # Mean labour force growth per country (repeat per row)
    data["Mean_Labour_Growth"] = data.groupby("country")["Labour_Force_Growth"].transform("mean")

    # GDP per worker (do NOT round yet to keep precision for model inversion)
    data["GDP_per_worker"] = data["Real_GDP"] / data["Labour_Force"]

    # Keep the most recent row per country with valid GDP_per_worker
    latest_idx = data.dropna(subset=["GDP_per_worker"]).groupby("country")["date"].idxmax()
    latest = data.loc[latest_idx].reset_index(drop=True)

    return latest

solow_df = load_and_process_data(START_DATE, END_DATE, INDICATORS)

# -----------------------
# Sidebar / Inputs
# -----------------------
with st.sidebar:
    st.subheader("Model parameters")
    alpha = st.number_input(
        "Capital share (α)",
        min_value=0.01, max_value=0.99, value=0.33, step=0.01, format="%.2f"
    )
    A = st.number_input("TFP level (A)", min_value=0.10, max_value=1000.0, value=1.00, step=0.10, format="%.2f")
    s = st.number_input("Savings rate (s)", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
    delta = st.number_input("Depreciation (δ)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
    T = st.number_input("Simulation periods (T)", min_value=1, max_value=2000, value=100, step=1)

# Country selection (top of main)
countries = solow_df["country"].sort_values().unique()
selected_country = st.selectbox("Select a country:", countries)

# Current country row
row = solow_df.loc[solow_df["country"] == selected_country].iloc[0]
y_data = float(row["GDP_per_worker"])                   # observed GDP per worker (latest)
n = float(row["Mean_Labour_Growth"]) if pd.notna(row["Mean_Labour_Growth"]) else 0.01
N0 = float(row["Labour_Force"])
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
def initial_k_from_output(y_per_worker, A, alpha):
    """Invert y = A * k^alpha  =>  k0 = (y/A)^(1/alpha)."""
    return (y_per_worker / A) ** (1.0 / alpha)

def solow_k_path(k0, A, alpha, s, delta, n, T):
    """k_{t+1} = [ s*A*k_t^α + (1-δ)k_t ] / (1+n)"""
    k = np.empty(T, dtype=float)
    k[0] = k0
    for t in range(1, T):
        k[t] = (s * A * (k[t-1] ** alpha) + (1.0 - delta) * k[t-1]) / (1.0 + n)
    return k

def lf_path(N0, n, T):
    """Labour force path with constant growth n."""
    return N0 * (1.0 + n) ** np.arange(T)

# -----------------------
# Build paths
# -----------------------
k0 = initial_k_from_output(y_data, A, alpha)
k_path = solow_k_path(k0, A, alpha, s, delta, n, T)

# Vectorized macro identities
y_path = A * (k_path ** alpha)          # output per (effective) worker
i_path = s * y_path                     # investment per worker
c_path = (1.0 - s) * y_path             # consumption per worker
w_path = (1.0 - alpha) * y_path         # wage (under Cobb-Douglas, competitive factor shares)

N_path = lf_path(N0, n, T)              # labour force
GDP_path = y_path * N_path              # total output

# -----------------------
# Headline & country panel
# -----------------------
st.markdown(f"**Latest data for {country_name} (year: {latest_year})**")
st.dataframe(
    pd.DataFrame({
        "Latest year": [latest_year],
        "Labour force": [N0],
        "Real GDP": [row["Real_GDP"]],
        "GDP per worker": [y_data],
        "Mean labour-force growth (n)": [n]
    }).T.rename(columns={0: "value"})
)

# -----------------------
# Plots (2 x 3 grid)
# -----------------------
time = np.arange(T)

def make_line(y, title, ylab):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode="lines", name=title))
    fig.update_layout(title=title, xaxis_title="Periods", yaxis_title=ylab, template="plotly_white")
    return fig

col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(make_line(k_path, f"Capital per worker (k) — {country_name}", "k"), use_container_width=True)
    st.plotly_chart(make_line(y_path, f"Output per worker (y) — {country_name}", "y"), use_container_width=True)
    st.plotly_chart(make_line(GDP_path, f"Total GDP — {country_name}", "GDP"), use_container_width=True)

with col_right:
    st.plotly_chart(make_line(i_path, f"Investment per worker (i) — {country_name}", "i"), use_container_width=True)
    st.plotly_chart(make_line(c_path, f"Consumption per worker (c) — {country_name}", "c"), use_container_width=True)
    st.plotly_chart(make_line(w_path, f"Wage per worker (w) — {country_name}", "w"), use_container_width=True)

# -----------------------
# Download results
# ----------------

paths_df = pd.DataFrame({
    "t": time,
    "k": k_path,
    "y": y_path,
    "i": i_path,
    "c": c_path,
    "w": w_path,
    "N": N_path,
    "GDP": GDP_path
})
st.download_button(
    "Download simulated paths (CSV)",
    data=paths_df.to_csv(index=False).encode("utf-8"),
    file_name=f"solow_paths_{country_name}.csv",
    mime="text/csv"
)
