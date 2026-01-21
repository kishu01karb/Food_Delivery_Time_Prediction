
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="🍔 Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

# ===================== STYLES =====================
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF4B4B;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    font-size: 2rem;
    font-weight: bold;
    color: #FF4B4B;
}
.help-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #2196F3;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown('<h1 class="main-header">🍔 Food Delivery Time Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Powered by XGBoost Machine Learning 🤖")

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("delivery_time_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        return model, encoders
    except Exception as e:
        st.error("❌ Model files not found or failed to load.")
        st.code(str(e))
        st.stop()

model, label_encoders = load_model()

# ===================== PRESETS =====================
if "preset" not in st.session_state:
    st.session_state.preset = "urban_lunch"

presets = {
    "urban_lunch": dict(distance=5.2, age=28, rating=4.5, multiple=2, hour=13, weekend=False),
    "late_night": dict(distance=3.1, age=25, rating=4.7, multiple=0, hour=23, weekend=True),
    "weekend_dinner": dict(distance=7.8, age=32, rating=4.3, multiple=1, hour=20, weekend=True)
}

preset = presets[st.session_state.preset]

# ===================== QUICK BUTTONS =====================
st.markdown("### 🎯 Quick Start Examples")
c1, c2, c3 = st.columns(3)
if c1.button("🏙️ Urban Lunch"): st.session_state.preset = "urban_lunch"
if c2.button("🌃 Late Night"): st.session_state.preset = "late_night"
if c3.button("🏡 Weekend Dinner"): st.session_state.preset = "weekend_dinner"

st.divider()

# ===================== INPUTS =====================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Details")

    distance = st.number_input("📍 Distance (km)", 0.1, 50.0, preset["distance"], 0.1)
    order_hour = st.slider("🕐 Order Hour", 0, 23, preset["hour"])
    is_peak = int(11 <= order_hour <= 14 or 18 <= order_hour <= 22)
    is_weekend = st.checkbox("📅 Weekend", preset["weekend"])

    age = st.slider("👤 Delivery Age", 18, 60, preset["age"])
    rating = st.slider("⭐ Rating", 1.0, 5.0, preset["rating"], 0.1)
    multiple = st.selectbox("📦 Multiple Deliveries", [0, 1, 2, 3], preset["multiple"])

    weather = st.selectbox("🌦 Weather", list(label_encoders["Weatherconditions"].classes_))
    traffic = st.selectbox("🚦 Traffic", list(label_encoders["Road_traffic_density"].classes_))
    order_type = st.selectbox("🍽 Order Type", list(label_encoders["Type_of_order"].classes_))
    vehicle = st.selectbox("🛵 Vehicle", list(label_encoders["Type_of_vehicle"].classes_))
    city = st.selectbox("🏙 City", list(label_encoders["City"].classes_))
    festival = st.selectbox("🎊 Festival", list(label_encoders["Festival"].classes_))

# ===================== INFO PANEL =====================
with col2:
    st.subheader("📊 Quick Info")

    st.success("🔥 Peak Hour" if is_peak else "✅ Off Peak")

    traffic_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Jam": "🔴"}
    st.info(f"🚦 {traffic_emoji.get(traffic, '⚪')} {traffic}")

# ===================== PREDICTION =====================
st.divider()

if st.button("🔮 Predict Delivery Time", type="primary", use_container_width=True):
    try:
        input_df = pd.DataFrame([{
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "Weatherconditions_encoded": label_encoders["Weatherconditions"].transform([weather])[0],
            "Road_traffic_density_encoded": label_encoders["Road_traffic_density"].transform([traffic])[0],
            "is_peak_hour": is_peak,
            "is_weekend": int(is_weekend),
            "delivery_distance_km": distance,
            "Type_of_order_encoded": label_encoders["Type_of_order"].transform([order_type])[0],
            "Type_of_vehicle_encoded": label_encoders["Type_of_vehicle"].transform([vehicle])[0],
            "Festival_encoded": label_encoders["Festival"].transform([festival])[0],
            "City_encoded": label_encoders["City"].transform([city])[0],
            "multiple_deliveries": multiple,
            "order_hour": order_hour
        }])

        prediction = model.predict(input_df)[0]

        st.balloons()
        st.markdown(f"<div class='prediction-box'>⏱ {prediction:.0f} minutes</div>", unsafe_allow_html=True)

        arrival = datetime.now() + pd.to_timedelta(prediction, unit="m")
        st.info(f"🕐 Estimated Arrival: {arrival.strftime('%I:%M %p')}")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))

# ===================== FOOTER =====================
st.divider()
st.markdown(
    "<center>🤖 XGBoost ML Model • Built with Streamlit</center>",
    unsafe_allow_html=True
)
