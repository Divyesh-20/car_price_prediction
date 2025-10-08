import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1️⃣ Load dataset, model & encoders
# -----------------------------
df_model = pd.read_csv("processed_car_dataset.csv")
features = ['brand', 'model', 'year', 'age', 'mileage', 'mileage_per_year', 
            'engine_size', 'fuel_type', 'transmission', 'condition', 
            'owner_count', 'color', 'body_type']
target = 'price_local'

model = joblib.load("car_price_xgb_model.pkl")
encoders = joblib.load("label_encoders.pkl")

cat_cols = ['brand', 'model', 'fuel_type', 'transmission', 'condition', 'color', 'body_type']

# Decode categorical for UI
df_ui = df_model.copy()
for col in cat_cols:
    df_ui[col] = encoders[col].inverse_transform(df_ui[col])

# -----------------------------
# 2️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: darkblue;'>🚗 Car Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter car details below to get an instant price estimate.</p>", unsafe_allow_html=True)
st.markdown("---")

# Columns layout
col1, col2 = st.columns(2)

with col1:
    selected_brand = st.selectbox("Select Brand", sorted(df_ui['brand'].unique()))
    available_models = df_ui[df_ui['brand'] == selected_brand]['model'].unique()
    selected_model = st.selectbox("Select Model", sorted(available_models))
    fuel_type = st.selectbox("Fuel Type", sorted(df_ui['fuel_type'].unique()))
    transmission = st.selectbox("Transmission", sorted(df_ui['transmission'].unique()))
    condition = st.selectbox("Condition", sorted(df_ui['condition'].unique()))
    color = st.selectbox("Color", sorted(df_ui['color'].unique()))
    body_type = st.selectbox("Body Type", sorted(df_ui['body_type'].unique()))

with col2:
    year = st.number_input("Year", min_value=1980, max_value=2025, value=2018)
    age = st.number_input("Age", min_value=0, max_value=30, value=5)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=50000)
    mileage_per_year = st.number_input("Mileage per year", min_value=0, max_value=100000, value=10000)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0)
    owner_count = st.number_input("Owner Count", min_value=0, max_value=5, value=1)

# -----------------------------
# 3️⃣ Prediction function
# -----------------------------
def predict_car_price(car_details: dict, conversion_rate=83):
    """
    Predicts car price from details dict, converts to INR, and formats for display.
    Returns None if an unknown categorical value is encountered.
    """
    df_input = pd.DataFrame([car_details])

    # Encode categorical columns
    for col in cat_cols:
        if df_input[col].iloc[0] not in encoders[col].classes_:
            st.warning(f"Unknown value '{df_input[col].iloc[0]}' for {col}.")
            return None
        df_input[col] = encoders[col].transform(df_input[col].values)

    # Predict price in USD
    pred_price_usd = model.predict(df_input)[0]

    # Convert to INR
    pred_price_inr = pred_price_usd * conversion_rate

    # Round and format price
    formatted_price = round(pred_price_inr, 0)

    return formatted_price


# -----------------------------
# 4️⃣ Predict button (Enhanced UI)
# -----------------------------
if st.button("💰 Predict Price"):
    car_details = {
        'brand': selected_brand,
        'model': selected_model,
        'fuel_type': fuel_type,
        'transmission': transmission,
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
    price = predict_car_price(car_details)
    
    if price is not None:
        st.markdown("---")

        # Dynamic price color based on range
        if price < 500000:
            price_color = "#16a34a"  # green
        elif price < 1500000:
            price_color = "#f59e0b"  # orange
        else:
            price_color = "#dc2626"  # red

        # Gradient header with dynamic color
        st.markdown(f"""
            <h2 style='text-align: center; color: {price_color}; 
                       font-weight: bold; font-size: 42px;'>
                🚗 Estimated Price: ₹{price:,}
            </h2>
        """, unsafe_allow_html=True)

        # Display brand logo (optional: you can map brand to image URLs)
#         brand_logos = {
#     'Audi': 'https://upload.wikimedia.org/wikipedia/commons/6/6f/Audi_logo_2016.png',
#     'BMW': 'https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg',
#     'Mercedes': 'https://upload.wikimedia.org/wikipedia/commons/9/90/Mercedes-Logo.svg',
#     'Toyota': 'https://upload.wikimedia.org/wikipedia/commons/9/9d/Toyota_logo.png',
#     'Honda': 'https://upload.wikimedia.org/wikipedia/commons/7/7b/Honda-logo.svg',
#     'Hyundai': 'https://upload.wikimedia.org/wikipedia/commons/3/3e/Hyundai_Motor_Company_logo.svg',
#     'Ford': 'https://upload.wikimedia.org/wikipedia/commons/3/3e/Ford_logo_flat.svg',
#     'Volkswagen': 'https://upload.wikimedia.org/wikipedia/commons/7/7f/Volkswagen_logo_2019.png',
#     'Skoda': 'https://upload.wikimedia.org/wikipedia/commons/d/d0/Škoda_logo.svg',
#     'Nissan': 'https://upload.wikimedia.org/wikipedia/commons/1/12/Nissan_logo.svg'
#     # Continue for other brands...
# }


        logo_url = brand_logos.get(selected_brand, None)
        if logo_url:
            st.markdown(f"<div style='text-align:center;'><img src='{logo_url}' width='120'/></div>", unsafe_allow_html=True)

        # Car info card with color-coded icons and glow effect
        st.markdown(
            f"""
            <div style='background-color:#000000; padding:25px; border-radius:15px; 
                        box-shadow: 0 8px 20px rgba(0,0,0,0.25); transition: all 0.3s ease-in-out;
                        hover: transform: scale(1.02);'>
                <h3 style='text-align:center; color:#1E3A8A;'>Car Details 📝</h3>
                <div style='display: flex; justify-content: space-between; flex-wrap: wrap;'>
                    <div style='flex: 45%; min-width: 200px;'>
                        <p>🚘 <b>Brand:</b> {selected_brand}</p>
                        <p>🏷️ <b>Model:</b> {selected_model}</p>
                        <p>⛽ <b>Fuel Type:</b> {fuel_type}</p>
                        <p>⚙️ <b>Transmission:</b> {transmission}</p>
                        <p>🟢 <b>Condition:</b> {condition}</p>
                        <p>🎨 <b>Color:</b> {color}</p>
                    </div>
                    <div style='flex: 45%; min-width: 200px;'>
                        <p>🏛️ <b>Body Type:</b> {body_type}</p>
                        <p>📅 <b>Year:</b> {year}</p>
                        <p>⏳ <b>Age:</b> {age} years</p>
                        <p>🛣️ <b>Mileage:</b> {mileage} km</p>
                        <p>📈 <b>Mileage/Year:</b> {mileage_per_year}</p>
                        <p>⚡ <b>Engine Size:</b> {engine_size} L</p>
                        <p>👤 <b>Owner Count:</b> {owner_count}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
