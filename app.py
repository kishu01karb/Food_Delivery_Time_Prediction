
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🍔 Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

# Custom CSS
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

# Title
st.markdown('<h1 class="main-header">🍔 Food Delivery Time Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Powered by XGBoost Machine Learning 🤖")

# Instructions
with st.expander("📖 **How to Use This App** - Click here for help!", expanded=False):
    st.markdown("""
    ### Welcome! This app predicts food delivery time. Here's how:
    
    **🎯 Quick Start Examples:**
    1. Click one of the preset buttons below
    2. Or enter your own details in the tabs
    3. Click "Predict" to get delivery time!
    
    **📍 Finding Coordinates:**
    - Open Google Maps
    - Right-click on restaurant/your location
    - Click the coordinates to copy them
    - Example cities:
      - Mumbai: 19.0760, 72.8777
      - Delhi: 28.7041, 77.1025
      - Bangalore: 12.9716, 77.5946
    
    **💡 Tips:**
    - Peak hours (11 AM-2 PM, 6 PM-10 PM) = longer wait
    - Check Google Maps for current traffic
    - Festival days usually have delays
    - All fields have help tooltips (hover over ℹ️)
    """)

# Quick presets
st.markdown("### 🎯 Quick Start: Try These Examples")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    if st.button("🏙️ Urban Lunch Rush", use_container_width=True):
        st.session_state.preset = "urban_lunch"
        
with col_p2:
    if st.button("🌃 Late Night Snack", use_container_width=True):
        st.session_state.preset = "late_night"
        
with col_p3:
    if st.button("🏡 Weekend Dinner", use_container_width=True):
        st.session_state.preset = "weekend_dinner"

st.write("---")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('delivery_time_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, encoders
    except:
        st.error("⚠️ Model files not found! Train the model first: `python delivery_predictor.py`")
        return None, None

model, label_encoders = load_model()

# Presets
if 'preset' not in st.session_state:
    st.session_state.preset = "urban_lunch"

presets = {
    "urban_lunch": {
        "distance": 5.2, "age": 28, "rating": 4.5, "multiple": 2,
        "weather": None, "traffic": None, "peak": True, "weekend": False,
        "hour": 13, "order_type": "Meal", "vehicle": "motorcycle", "city": "Metropolitan",
        "festival": None
    },
    "late_night": {
        "distance": 3.1, "age": 25, "rating": 4.7, "multiple": 0,
        "weather": None, "traffic": None, "peak": False, "weekend": True,
        "hour": 23, "order_type": "Snack", "vehicle": "scooter", "city": "Urban",
        "festival": None
    },
    "weekend_dinner": {
        "distance": 7.8, "age": 32, "rating": 4.3, "multiple": 1,
        "weather": None, "traffic": None, "peak": True, "weekend": True,
        "hour": 20, "order_type": "Meal", "vehicle": "electric_scooter", "city": "Semi-Urban",
        "festival": None
    }
}

preset = presets[st.session_state.preset]

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Details")
    
    tab1, tab2, tab3 = st.tabs(["📍 Distance & Time", "🚗 Delivery Info", "🌦️ Conditions"])
    
    with tab1:
        st.markdown("""
        <div class="help-box">
        💡 <strong>Don't know exact distance?</strong><br>
        Estimate: Within neighborhood (1-3 km) | Same area (3-7 km) | Far (7+ km)
        </div>
        """, unsafe_allow_html=True)
        
        distance = st.number_input(
            "📍 Delivery Distance (km)",
            min_value=0.1,
            max_value=50.0,
            value=float(preset["distance"]),
            step=0.1,
            help="Straight-line distance from restaurant to you. Use Google Maps for accuracy!"
        )
        
        order_hour = st.slider(
            "🕐 Order Hour (24-hour format)",
            min_value=0,
            max_value=23,
            value=int(preset["hour"]),
            help="What hour will you order? Peak: 11-14 (lunch) & 18-22 (dinner)"
        )
        
        is_peak = 1 if (11 <= order_hour <= 14 or 18 <= order_hour <= 22) else 0
        is_weekend = st.checkbox(
            "📅 Weekend? (Saturday/Sunday)",
            value=bool(preset["weekend"]),
            help="Different patterns on weekends!"
        )
    
    with tab2:
        col_a, col_b = st.columns(2)
        
        with col_a:
            delivery_age = st.slider(
                "👤 Delivery Person Age",
                18, 60, int(preset["age"]),
                help="Typical: 20-40 years. Don't worry if you don't know - use default!"
            )
            
            delivery_rating = st.slider(
                "⭐ Rating (1-5)",
                1.0, 5.0, float(preset["rating"]), 0.1,
                help="Most partners have 4.0-4.9 ratings"
            )
        
        with col_b:
            multiple_deliveries = st.selectbox(
                "📦 Multiple Orders?",
                [0, 1, 2, 3],
                index=int(preset["multiple"]),
                help="0=Only your order (fastest!) | 1-3=Shared delivery"
            )
            
            # Get available order types
            if label_encoders and 'Type_of_order' in label_encoders:
                available_order_types = list(label_encoders['Type_of_order'].classes_)
            else:
                available_order_types = ["Snack", "Meal", "Drinks", "Buffet"]
            
            type_of_order = st.selectbox(
                "🍽️ Order Type",
                options=available_order_types,
                help="Snack=Quick items | Meal=Full course | Buffet=Large order"
            )
        
        # Get available vehicle types
        if label_encoders and 'Type_of_vehicle' in label_encoders:
            available_vehicles = list(label_encoders['Type_of_vehicle'].classes_)
        else:
            available_vehicles = ["motorcycle", "scooter", "electric_scooter"]
        
        type_of_vehicle = st.selectbox(
            "🛵 Vehicle Type",
            options=available_vehicles,
            help="Motorcycle=Fastest | Scooter=Moderate | Electric=City speed"
        )
        
        # Get available city types
        if label_encoders and 'City' in label_encoders:
            available_cities = list(label_encoders['City'].classes_)
        else:
            available_cities = ["Metropolitan", "Urban", "Semi-Urban"]
        
        city = st.selectbox(
            "🏙️ City Type",
            options=available_cities,
            help="Metro=Big cities | Urban=Tier-2 | Semi-Urban=Towns"
        )
    
    with tab3:
        st.markdown("""
        <div class="help-box">
        🌦️ <strong>Check current conditions:</strong><br>
        Weather: Look outside or check weather app<br>
        Traffic: Open Google Maps to see live traffic colors
        </div>
        """, unsafe_allow_html=True)
        
        # Get available weather options from the encoder
        if label_encoders and 'Weatherconditions' in label_encoders:
            available_weather = list(label_encoders['Weatherconditions'].classes_)
        else:
            available_weather = ["Sunny", "Cloudy", "Fog", "Stormy"]
        
        weather = st.selectbox(
            "🌦️ Weather",
            options=available_weather,
            help="Clear=Fast | Rain/Storm=Slow"
        )
        
        # Get available traffic options from the encoder
        if label_encoders and 'Road_traffic_density' in label_encoders:
            available_traffic = list(label_encoders['Road_traffic_density'].classes_)
        else:
            available_traffic = ["Low", "Medium", "High", "Jam"]
        
        traffic = st.selectbox(
            "🚦 Traffic",
            options=available_traffic,
            help="Check Google Maps: Green=Low | Yellow=Medium | Red=High/Jam"
        )
        
        # Get available festival options
        if label_encoders and 'Festival' in label_encoders:
            available_festival = list(label_encoders['Festival'].classes_)
        else:
            available_festival = ["No", "Yes"]
        
        festival = st.selectbox(
            "🎊 Festival Today?",
            options=available_festival,
            help="Festivals = More orders = Longer waits"
        )

with col2:
    st.subheader("📊 Quick Info")
    
    if is_peak:
        st.error("🔥 **PEAK HOUR!**\n\nExpect delays")
    else:
        st.success("✅ **Off-Peak**\n\nGood timing!")
    
    # Distance indicator
    if distance < 2:
        st.metric("📍 Distance", f"{distance:.1f} km", "🟢 Very Close")
    elif distance < 5:
        st.metric("📍 Distance", f"{distance:.1f} km", "🟡 Moderate")
    else:
        st.metric("📍 Distance", f"{distance:.1f} km", "🔴 Far")
    
    # Delivery person
    st.metric("👤 Age", f"{delivery_age} years")
    rating_stars = "⭐" * int(delivery_rating)
    st.metric("⭐ Rating", f"{delivery_rating}/5.0")
    
    # Traffic
    traffic_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Jam": "🔴"}
    traffic_clean =str(traffic).strip().title()
    st.info(f"🚦 {traffic_emoji.get(traffic_clean,'⚪')} {traffic_clean}")
    
    # Weather
    weather_emoji = {"Sunny": "☀️", "Cloudy": "☁️", "Fog": "🌫️", 
                     "Sandstorms": "🌪️", "Stormy": "⛈️", "Windy": "💨",
                     "Jam": "🌫️"}  # Handle any weather
    st.info(f"🌦️ {weather_emoji.get(weather, '🌦️')} {weather}")

st.write("---")

# Prediction button
if st.button("🔮 Predict Delivery Time", type="primary", use_container_width=True):
    if model and label_encoders:
        try:
            # Prepare input
            input_data = {
                'Delivery_person_Age': delivery_age,
                'Delivery_person_Ratings': delivery_rating,
                'Weatherconditions_encoded': label_encoders['Weatherconditions'].transform([weather])[0],
                'Road_traffic_density_encoded': label_encoders['Road_traffic_density'].transform([traffic])[0],
                'is_peak_hour': is_peak,
                'is_weekend': 1 if is_weekend else 0,
                'delivery_distance_km': distance,
                'Type_of_order_encoded': label_encoders['Type_of_order'].transform([type_of_order])[0],
                'Type_of_vehicle_encoded': label_encoders['Type_of_vehicle'].transform([type_of_vehicle])[0],
                'Festival_encoded': label_encoders['Festival'].transform([festival])[0],
                'City_encoded': label_encoders['City'].transform([city])[0],
                'multiple_deliveries': multiple_deliveries,
                'order_hour': order_hour
            }
            
            input_df = pd.DataFrame([input_data])
            predicted_time = model.predict(input_df)[0]
            
            st.balloons()
            
            st.markdown("### 🎯 Prediction Result")
            st.markdown(f'<div class="prediction-box">⏱️ {predicted_time:.0f} minutes</div>', 
                       unsafe_allow_html=True)
            
            # Context
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                if predicted_time < 20:
                    st.success("⚡ **Super Fast!**")
                elif predicted_time < 30:
                    st.info("🚀 **Good Time**")
                elif predicted_time < 45:
                    st.warning("🐢 **Takes a While**")
                else:
                    st.error("⏰ **Long Wait**")
            
            with col_y:
                now = datetime.now()
                arrival = now.replace(
                    hour=(now.hour + int(predicted_time // 60)) % 24,
                    minute=(now.minute + int(predicted_time % 60)) % 60
                )
                st.info(f"🕐 **Arrival**\n\n{arrival.strftime('%I:%M %p')}")
            
            with col_z:
                buffer = predicted_time * 0.15
                st.info(f"⏰ **With Buffer**\n\n{predicted_time + buffer:.0f} min")
            
            # Insights
            st.write("---")
            st.subheader("💡 Factors Affecting Your Delivery")
            
            insights = []
            if is_peak:
                insights.append("🔥 Peak hour - High demand")
            if traffic in ["High", "Jam"]:
                insights.append("🚦 Heavy traffic slowing delivery")
            if weather in ["Stormy", "Sandstorms"]:
                insights.append("🌧️ Bad weather affecting speed")
            if festival == "Yes":
                insights.append("🎊 Festival day - More orders")
            if multiple_deliveries > 1:
                insights.append(f"📦 {multiple_deliveries} shared orders")
            if distance > 5:
                insights.append(f"📍 Long distance: {distance:.1f} km")
            elif distance < 2:
                insights.append(f"📍 Very close: {distance:.1f} km")
            
            for insight in insights:
                st.write(f"• {insight}")
            
            if not insights:
                st.success("✅ Perfect conditions for fast delivery!")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.error("⚠️ Model not loaded! Run `python delivery_predictor.py` first.")

# Footer
st.write("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by XGBoost Machine Learning</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.subheader("ℹ️ About")
    st.write("""
    This app uses **Machine Learning** to predict delivery times!
    
    **Features:**
    - 📍 Distance calculation
    - ⏰ Peak hour detection
    - 🚦 Traffic conditions
    - 🌦️ Weather impact
    - 👤 Driver experience
    - 🎊 Festival effects
    
    **Model:** XGBoost Regression
    **Accuracy:** ±3-5 minutes
    """)
    
    if model:
        st.success("✅ Model Ready")
    else:
        st.error("❌ Train model first!")
    
    # Show available options for debugging
    with st.expander("🔍 Available Options", expanded=False):
        if label_encoders:
            st.write("**Weather Options:**")
            if 'Weatherconditions' in label_encoders:
                st.write(list(label_encoders['Weatherconditions'].classes_))
            
            st.write("**Traffic Options:**")
            if 'Road_traffic_density' in label_encoders:
                st.write(list(label_encoders['Road_traffic_density'].classes_))
            
            st.write("**Order Types:**")
            if 'Type_of_order' in label_encoders:
                st.write(list(label_encoders['Type_of_order'].classes_))
            
            st.write("**Vehicle Types:**")
            if 'Type_of_vehicle' in label_encoders:
                st.write(list(label_encoders['Type_of_vehicle'].classes_))
            
            st.write("**City Types:**")
            if 'City' in label_encoders:
                st.write(list(label_encoders['City'].classes_))
        else:
            st.write("Load encoders to see options")
