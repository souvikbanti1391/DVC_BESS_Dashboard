# =========================================================
# ⚡ DVC – Interactive BESS Financial Dashboard (Streamlit Cloud)
# Permanent deployment version – no Colab, no ngrok
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="DVC BESS Financial Dashboard", layout="wide")

# Header
c1, c2 = st.columns([1, 6])
with c1:
    try:
        st.image("assets/dvc_logo.png", width=90)
    except Exception:
        st.warning("DVC logo not found (place 'dvc_logo.png' in assets/)")
with c2:
    st.title("⚡ DVC – Interactive BESS Financial Dashboard")
st.caption("Forecasting • Arbitrage Simulation • Financial Evaluation")
st.markdown("---")

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("1. Upload & Forecast Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV (datetime + price)", type=["csv"])
dayfirst = st.sidebar.checkbox("Parse dates as DD/MM/YYYY", value=False)
default_method = st.sidebar.selectbox("Default Forecast Model", ["SARIMAX (auto)", "Hourly Mean"], index=0)

st.sidebar.header("2. BESS & Financial Parameters")
P = st.sidebar.number_input("Rated Power (MW)", value=10.0, min_value=0.1)
H = st.sidebar.selectbox("Duration (hours)", [1, 2, 4, 6, 8], index=2)
cycles = st.sidebar.selectbox("Cycles per day", [1, 2, 4], index=0)
eff = st.sidebar.slider("Round-trip Efficiency (%)", 60, 100, 90) / 100
capex_mwh = st.sidebar.number_input("CAPEX (₹/MWh)", value=1700000.0)
fix_om_pct = st.sidebar.slider("Fixed O&M (%CAPEX/year)", 0.0, 10.0, 2.0) / 100
var_om_mwh = st.sidebar.number_input("Variable O&M (₹/MWh)", value=50.0)
cycles_EOL = st.sidebar.number_input("Cycles to EOL", value=3000)
disc = st.sidebar.number_input("Discount Rate (%)", value=8.0) / 100
yrs = st.sidebar.number_input("Project Life (years)", value=15, min_value=1)
salv = st.sidebar.slider("Salvage Value (%CAPEX)", 0.0, 50.0, 0.0) / 100

# ---------------------------------------------------------
# Helper function to detect datetime/price columns
# ---------------------------------------------------------
def detect_cols(df):
    tcol = next((c for c in df.columns if any(x in c.lower() for x in ["time","date","timestamp","block"])), None)
    pcol = next((c for c in df.columns if any(x in c.lower() for x in ["price","rs","₹","inr","mcp"])), None)
    return tcol, pcol

# ---------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload CSV to begin (must include datetime and price columns).")
    st.stop()

df_raw = pd.read_csv(uploaded_file, dtype=str)
tcol, pcol = detect_cols(df_raw)
if not tcol or not pcol:
    st.error("❌ Couldn't detect timestamp or price column. Rename appropriately.")
    st.stop()

df_raw[tcol] = pd.to_datetime(df_raw[tcol], errors="coerce", dayfirst=dayfirst)
df_raw[pcol] = df_raw[pcol].astype(str).str.replace(",","").str.replace("₹","").str.replace("Rs","").str.replace("INR","")
df_raw["price"] = pd.to_numeric(df_raw[pcol], errors="coerce")
df = df_raw.dropna(subset=[tcol,"price"]).rename(columns={tcol:"timestamp"}).sort_values("timestamp").set_index("timestamp")

if df["price"].median() > 100:
    df["price"] /= 1000
    st.info("Detected ₹/MWh; converted to ₹/kWh")

# ---------------------------------------------------------
# Historical range selector
# ---------------------------------------------------------
st.subheader("1️⃣ Historical Price (1–7 day range)")
last_d = df.index.max().date()
first_d = last_d - timedelta(days=6)
s, e = st.date_input("Select range", (first_d, last_d), min_value=first_d, max_value=last_d)
mask = (df.index.date >= s) & (df.index.date <= e)
hist = df.loc[mask] if not df.loc[mask].empty else df.last("168H")

fig_h = px.line(hist.reset_index(), x="timestamp", y="price", title="Historical Prices", labels={"price":"₹/kWh"})
fig_h.update_xaxes(tickformat="%d %b\n%H:%M")
st.plotly_chart(fig_h, use_container_width=True)

# ---------------------------------------------------------
# Forecast (SARIMAX or Hourly Mean)
# ---------------------------------------------------------
st.subheader("2️⃣ Forecast (next 7 days)")

f_start = df.index.max() + timedelta(hours=1)
idx = pd.date_range(f_start, periods=24*7, freq="H")
method = st.selectbox("Forecast Method", ["SARIMAX (auto)", "Hourly Mean"],
                      index=0 if default_method == "SARIMAX (auto)" else 1)

try:
    if method.startswith("SARIMAX"):
        seas = 168 if len(df) > 336 else 24
        model = SARIMAX(df["price"], order=(1,1,1), seasonal_order=(1,0,1,seas),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = model.get_forecast(steps=len(idx))
        forecast = pd.Series(fc.predicted_mean, index=idx)
        ci = fc.conf_int(alpha=0.05); ci.index = idx
        forecast_ci = ci
        used = f"SARIMAX (season={seas})"
    else:
        hm = df["price"].groupby(df.index.hour).mean()
        forecast = pd.Series([hm.get(ts.hour, df["price"].mean()) for ts in idx], index=idx)
        std = df["price"].groupby(df.index.hour).std().fillna(df["price"].std())
        forecast_ci = pd.DataFrame({
            "lower": forecast - [1.96*std.get(ts.hour, df["price"].std()) for ts in idx],
            "upper": forecast + [1.96*std.get(ts.hour, df["price"].std()) for ts in idx]
        }, index=idx)
        used = "Hourly Mean"

    st.markdown(f"**Model:** {used}")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecast"))
    fig_f.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci["upper"], mode="lines",
                               line_color="rgba(0,0,0,0)", showlegend=False))
    fig_f.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci["lower"], mode="lines",
                               fill="tonexty", fillcolor="rgba(0,100,80,0.1)", showlegend=False))
    fig_f.update_xaxes(tickformat="%d %b\n%H:%M")
    st.plotly_chart(fig_f, use_container_width=True)
except Exception as e:
    st.error(f"Forecast failed: {e}")
    forecast = df["price"]

# ---------------------------------------------------------
# Arbitrage window recommendation
# ---------------------------------------------------------
st.subheader("3️⃣ Arbitrage Hours (Recommended & Manual)")

df_f = forecast.to_frame("price"); df_f["hour"] = df_f.index.hour
avg = df_f.groupby("hour")["price"].mean().sort_values()
n = st.slider("Hours to use per day", 1, 6, 2)
ch = list(avg.index[:n]); dh = list(avg.index[-n:])[::-1]
col1, col2 = st.columns(2)
col1.write("Charge hours:"); col1.write(ch)
col2.write("Discharge hours:"); col2.write(dh)
spread = df_f[df_f["hour"].isin(dh)]["price"].mean() - df_f[df_f["hour"].isin(ch)]["price"].mean()
st.markdown(f"**Avg Spread:** ₹{spread:.3f}/kWh")

charge = st.multiselect("Manual Charge hours (override)", list(range(24)), default=ch)
disch = st.multiselect("Manual Discharge hours (override)", list(range(24)), default=dh)

# ---------------------------------------------------------
# Financial model
# ---------------------------------------------------------
st.subheader("4️⃣ Financial Model")

Pkw = P*1000
E = P*H
Ekwh = E*1000
cycle_kwh = min(Pkw*H, Ekwh)
thr_day = cycle_kwh * cycles
price_c = df_f[df_f["hour"].isin(charge)]["price"].mean()
price_d = df_f[df_f["hour"].isin(disch)]["price"].mean()
if np.isnan(price_c): price_c = df["price"].min()
if np.isnan(price_d): price_d = df["price"].max()

deliv = cycle_kwh * eff
cost = cycle_kwh * price_c
rev = deliv * price_d
net = rev - cost
annual_net = net * cycles * 365

capex = E * capex_mwh
ann_factor = (disc*(1+disc)**yrs)/(((1+disc)**yrs)-1)
ann_capex = capex * ann_factor
fix_om = capex * fix_om_pct
var_om = var_om_mwh * (thr_day/1000) * 365
life_thr = E * cycles_EOL
deg = (capex/life_thr) * (thr_day/1000) * 365 if life_thr>0 else 0
ann_cf = annual_net - (fix_om + var_om + deg)
npv = sum((ann_cf if i>0 else -capex)/((1+disc)**i) for i in range(yrs+1))
payback = capex/ann_cf if ann_cf>0 else None

c1,c2,c3 = st.columns(3)
c1.metric("Annual Net (₹)", f"{annual_net:,.0f}")
c2.metric("CAPEX (₹)", f"{capex:,.0f}")
c3.metric("NPV (₹)", f"{npv:,.0f}")
st.write(f"Payback (years): {payback:.2f}" if payback else "Payback > project life")
st.write(f"O&M Fixed ₹{fix_om:,.0f}, Var ₹{var_om:,.0f}, Degradation ₹{deg:,.0f}")

# Sensitivity
sp_base = price_d - price_c
spreads = np.linspace(sp_base*0.8, sp_base*1.2, 9)
npvs = []
for sp in spreads:
    scale = sp/sp_base if sp_base!=0 else 1
    new_cf = ann_cf * scale
    cf = [-capex] + [new_cf]*yrs
    npvs.append(sum(cf[i]/((1+disc)**i) for i in range(len(cf))))
figs = px.line(x=spreads, y=npvs, labels={"x":"Spread ₹/kWh","y":"NPV ₹"}, title="NPV vs Spread ±20%")
st.plotly_chart(figs, use_container_width=True)

st.markdown("---")
st.caption("© 2025 Damodar Valley Corporation | System Planning & Energy Storage Studies")
