# car_price_dashboard_pro.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import plotly.express as px
import time

# -----------------------------
# 1️⃣ Load Data & Model
# -----------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("data/dataset.csv")
    for col in ['brand', 'model', 'fuel_type', 'transmission', 'condition', 'color', 'body_type']:
        df[col] = df[col].astype(str).str.strip().str.title()
    model = joblib.load("car_price_xgb_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    brand_model_map = df.groupby('brand')['model'].apply(lambda x: sorted(x.unique())).to_dict()
    return df, model, encoders, brand_model_map

df, model, encoders, brand_model_map = load_data()
cat_cols = ['brand', 'model', 'fuel_type', 'transmission', 'condition', 'color', 'body_type']
features = ['brand', 'model', 'fuel_type', 'transmission', 'condition',
            'color', 'body_type', 'year', 'age', 'mileage',
            'mileage_per_year', 'engine_size', 'owner_count']

# -----------------------------
# 2️⃣ Page Config + Theme CSS
# -----------------------------
st.set_page_config(page_title="🚗 Car Price Dashboard", layout="wide")

st.markdown("""
<style>
/* ----------------- Global ----------------- */
.stApp { background: radial-gradient(circle at top left, #1a0033, #0d001a); color: #EAEAEA; }
h1 { color: #CFA0FF; text-shadow: 2px 2px 8px #00000088; text-align: center; }
h3, h4 { color: #B38BFA; text-shadow: 1px 1px 4px #00000066; }

/* Sidebar */
section[data-testid="stSidebar"] {
    backdrop-filter: blur(8px);
    background: rgba(25, 0, 51, 0.7);
    border-right: 2px solid #5D3FD3;
    color: #fff;
}

/* Tabs */
div[data-baseweb="tab-list"] button {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #CFA0FF !important;
    border-radius: 12px !important;
    margin-right: 6px !important;
    font-weight: 600;
}
div[data-baseweb="tab-list"] button:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    color: #fff !important;
}

/* Metrics */
[data-testid="stMetricValue"] { color: #B38BFA !important; font-weight: bold; font-size: 22px; text-shadow: 1px 1px 4px #00000066; }

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(90deg, #5D3FD3, #9D4EDD);
    color: white; border-radius: 10px; height: 3em; width: 100%; font-weight: 600;
    transition: 0.3s; font-size: 16px;
}
button[kind="primary"]:hover { background: linear-gradient(90deg, #9D4EDD, #5D3FD3); transform: scale(1.02); }

/* Inputs */
.stSelectbox, .stNumberInput { background-color: #160030 !important; border-radius: 8px !important; }

/* Plot background */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* Divider */
hr { border-color: #5D3FD3; }

/* Popup prediction */
.popup-card {
    position: fixed; top: 40%; left: 50%; transform: translate(-50%, -50%);
    background: rgba(40,0,80,0.95); border: 2px solid #9D4EDD;
    padding: 25px 45px; border-radius: 15px; text-align: center;
    box-shadow: 0px 0px 25px #5D3FD3AA; z-index: 9999; animation: fadeIn 0.6s ease-in-out;
}
.popup-price { color: #CFA0FF; font-size: 2rem; font-weight: 700; margin-bottom: 10px; }
.popup-info { color:#aaa; font-size:14px; }
@keyframes fadeIn { from {opacity:0; transform:translate(-50%,-45%);} to {opacity:1; transform:translate(-50%,-50%);} }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3️⃣ Header
# -----------------------------
st.markdown("<h1>🚗 Car Price Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# 4️⃣ Sidebar Filters
# -----------------------------
st.sidebar.header("⚙️ Filter Your Selection")
selected_brand = st.sidebar.selectbox("Select Brand", sorted(brand_model_map.keys()))
available_models = brand_model_map[selected_brand]
selected_model = st.sidebar.selectbox("Select Model", available_models)

available_fuels = sorted(df[(df['brand']==selected_brand) & (df['model']==selected_model)]['fuel_type'].unique())
selected_fuel = st.sidebar.selectbox("Fuel Type", available_fuels)

available_trans = sorted(df[(df['brand']==selected_brand) & 
                            (df['model']==selected_model) & 
                            (df['fuel_type']==selected_fuel)]['transmission'].unique())

selected_transmission = st.sidebar.selectbox("Transmission", available_trans)

year_range = st.sidebar.slider("Year Range", int(df['year'].min()), int(df['year'].max()), (2010, 2023))

df_filtered = df[
    (df['brand']==selected_brand) &
    (df['model']==selected_model) &
    (df['fuel_type']==selected_fuel) &
    (df['transmission']==selected_transmission) &
    (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
]

# -----------------------------
# 5️⃣ Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "📈 Model Comparison", "💰 Price Prediction"])

# ---- Tab 1: Market Overview ----
with tab1:
    st.markdown(f"### Market Overview: **{selected_brand} {selected_model}**")
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    col1.metric("Total Cars", len(df_filtered))
    col2.metric("Min Price (₹)", int(df_filtered['price_local'].min() if not df_filtered.empty else 0))
    col3.metric("Max Price (₹)", int(df_filtered['price_local'].max() if not df_filtered.empty else 0))
    col4.metric("Avg Price (₹)", int(df_filtered['price_local'].mean() if not df_filtered.empty else 0))
    col5.metric("Avg Mileage (km)", int(df_filtered['mileage'].mean() if not df_filtered.empty else 0))

    st.markdown("### 💵 Price Distribution")
    fig_price_dist = px.histogram(
        df_filtered, x='price_local', nbins=25, title='Distribution of Prices',
        color_discrete_sequence=px.colors.sequential.Purples
    )
    fig_price_dist.update_traces(marker_line_width=1, marker_line_color='rgba(0,0,0,1)')
    fig_price_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                 plot_bgcolor='rgba(0,0,0,0)',
                                 font=dict(color='#CFA0FF'))
    fig_price_dist.update_traces(hovertemplate='₹%{x:,.0f}<extra></extra>')
    st.plotly_chart(fig_price_dist, use_container_width=True, config={'displayModeBar': False})

    st.markdown("### 📆 Historical Price Trend")
    if not df_filtered.empty:
        df_trend = df_filtered.groupby('year')['price_local'].mean().reset_index()
        fig_trend = px.line(df_trend, x='year', y='price_local', markers=True,
                            title='Average Price by Year',
                            color_discrete_sequence=['#A06CD5'])
        fig_trend.update_traces(line=dict(width=4))
        fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#CFA0FF'))
        fig_trend.update_traces(hovertemplate='Year %{x}: ₹%{y:,.0f}<extra></extra>')
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

# ---- Tab 2: Model Comparison ----
with tab2:
    st.markdown(f"### 💹 {selected_brand} Model Price Comparison")
    df_compare = df[df['brand']==selected_brand].groupby('model')['price_local'].mean().reset_index()
    fig_compare = px.bar(
        df_compare, x='model', y='price_local', text='price_local', color='model',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_compare.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside',
                              marker_line_width=1, marker_line_color='rgba(255,255,255,0.3)')
    fig_compare.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#CFA0FF'))
    fig_compare.update_traces(hovertemplate='Model %{x}: ₹%{y:,.0f}<extra></extra>')
    st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': False})

# ---- Tab 3: Prediction ----
with tab3:
    st.markdown("### 🎯 Predict Your Car Price")
    with st.form("car_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            condition = st.selectbox("Condition 🏷️", sorted(encoders['condition'].classes_))
            color = st.selectbox("Color 🎨", sorted(encoders['color'].classes_))
            body_type = st.selectbox("Body Type 🚙", sorted(encoders['body_type'].classes_))
        with col2:
            year = st.number_input("Year of Manufacture 📅", 1990, datetime.now().year, 2016)
            age = datetime.now().year - year
            st.metric("Calculated Age", f"{age} years")
            mileage = st.number_input("Usage (km) 🛣️", 0, 1000000, 50000)
            mileage_per_year = mileage / (age + 1)
            st.metric("Usage per Year", f"{int(mileage_per_year)} km/year")
            engine_size = st.number_input("Engine Size (L) 🏎️", 0.0, 10.0, 1.6, 0.1)
            owner_count = st.number_input("Previous Owners 👤", 0, 10, 1)
        submitted = st.form_submit_button("💰 Predict Price")

    if submitted:
        car_details = {
            'brand': selected_brand,
            'model': selected_model,
            'fuel_type': selected_fuel,
            'transmission': selected_transmission,
            'condition': condition,
            'color': color,
            'body_type': body_type,
            'year': year,
            'age': age,
            'mileage': mileage,
            'mileage_per_year': mileage_per_year,
            'engine_size': engine_size,
            'owner_count': owner_count
        }
        for col in cat_cols:
            car_details[col] = encoders[col].transform([str(car_details[col]).strip().title()])[0]
        df_input = pd.DataFrame([car_details])[features]
        predicted_price = model.predict(df_input)[0]

        # Popup style prediction
        st.markdown(f"""
        <div class="popup-card">
            <div class="popup-price">💰 Estimated Price: ₹ {round(predicted_price):,}</div>
            <div class="popup-info">Based on real-world market analysis 🚀</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
