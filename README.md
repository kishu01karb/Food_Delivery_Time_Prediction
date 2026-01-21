# 🍔 Food Delivery Time Prediction

A machine learning project that predicts food delivery times using **XGBoost** algorithm. Includes a beautiful **Streamlit web interface** for easy predictions!

Project is live at :https://fooddeliverytimeprediction-5qv3dbbdxcmnz6wthr8vsy.streamlit.app/#enter-details
---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project predicts **how long food delivery will take** based on various factors like:
- 📍 Distance between restaurant and customer
- ⏰ Time of order (peak vs off-peak hours)
- 🚦 Traffic conditions
- 🌦️ Weather conditions
- 👤 Delivery person age and ratings
- 🛵 Type of vehicle used
- 🎊 Festival/holiday status

Perfect for delivery apps like **Uber Eats, DoorDash, Swiggy, or Zomato**!

---

## ✨ Features

### 🤖 Machine Learning Model
- **Algorithm:** XGBoost Regressor
- **Accuracy:** Predictions typically within ±3-5 minutes
- **Features:** 13 engineered features including distance calculation using Haversine formula

### 🌐 Web Interface (Streamlit)
- 📱 **Mobile-friendly** responsive design
- 🎯 **Quick Start Presets** - Try example scenarios instantly
- 💡 **Contextual Help** - Tooltips and guides throughout
- 📊 **Live Metrics** - Real-time calculations and indicators
- 🎨 **Beautiful UI** - Color-coded status indicators
- 💭 **Smart Insights** - Explains factors affecting delivery time

---

## 🖼️ Demo

### Main Interface
```
🍔 Food Delivery Time Predictor
Powered by XGBoost Machine Learning 🤖

[Quick Start: Urban Lunch Rush] [Late Night Snack] [Weekend Dinner]

📝 Enter Details
├── 📍 Distance & Time
│   ├── Distance: 5.2 km
│   ├── Order Hour: 13 (1 PM)
│   └── Weekend: ☐
├── 🚗 Delivery Info
│   ├── Age: 28 years
│   ├── Rating: 4.5/5
│   └── Vehicle: motorcycle
└── 🌦️ Conditions
    ├── Weather: Sunny
    ├── Traffic: High
    └── Festival: No

[🔮 Predict Delivery Time]

Result: ⏱️ 28 minutes
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/food-delivery-prediction.git
cd food-delivery-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python --version  # Should be 3.8+
streamlit --version
```

---

## 📊 Usage

### 1️⃣ Train the Model

First, train the XGBoost model on your dataset:

```bash
 delivery_predictor.py
```

**Output:**
```
📦 Loading data...
🧹 Cleaning data...
🔧 Converting data types...
🛠️ Engineering features...
🤖 Training XGBoost model...

=================================================
🎯 MODEL PERFORMANCE
=================================================
📊 MAE: 3.24 minutes
📊 RMSE: 4.51 minutes
📊 R² Score: 0.847
=================================================

✅ All done! Model saved and ready to use! 🎉
```

This creates two files:
- `delivery_time_model.pkl` - Your trained model
- `label_encoders.pkl` - Encoders for categorical variables

### 2️⃣ Launch the Web App

Start the Streamlit web interface:

```bash
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501`

### 3️⃣ Make Predictions

**Option A: Use Quick Presets**
1. Click "Urban Lunch Rush", "Late Night Snack", or "Weekend Dinner"
2. Click "Predict" to see results

**Option B: Enter Custom Details**
1. Fill in the tabs:
   - 📍 Distance & Time
   - 🚗 Delivery Info  
   - 🌦️ Conditions
2. Click "🔮 Predict Delivery Time"
3. View your prediction with insights!

---

## 📁 Project Structure

```
food-delivery-prediction/
│
├── data/
│   └── food_delivery_time.csv          # Dataset
│
├── delivery_predictor.py               # Model training script
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── delivery_time_model.pkl            # Trained model (generated)
└── label_encoders.pkl                 # Encoders (generated)
```

---

## 📈 Model Performance

### Metrics
- **MAE (Mean Absolute Error):** ~3-5 minutes
  - On average, predictions are off by 3-5 minutes
- **RMSE (Root Mean Squared Error):** ~4-6 minutes
  - Penalizes larger errors more heavily
- **R² Score:** ~0.80-0.85
  - Model explains 80-85% of variance in delivery times

### Feature Importance
Top factors affecting delivery time:
1. 📍 **Distance** (35%) - Most important factor
2. 🚦 **Traffic Density** (20%)
3. ⏰ **Peak Hour** (15%)
4. 🌦️ **Weather** (12%)
5. 👤 **Delivery Person Rating** (10%)
6. 🛵 **Vehicle Type** (8%)

---

## 🔬 How It Works

### Data Pipeline

```
Raw Data → Cleaning → Feature Engineering → Model Training → Predictions
```

### 1. Data Cleaning
- Handle missing values
- Convert data types
- Remove duplicates
- Clean target variable

### 2. Feature Engineering

**Created Features:**
- `order_hour` - Hour of the day (0-23)
- `is_peak_hour` - Peak times: 11 AM-2 PM, 6 PM-10 PM
- `is_weekend` - Saturday/Sunday indicator
- `delivery_distance_km` - Haversine distance calculation

**Formula for Distance:**
```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    # Convert to radians and calculate great circle distance
    ...
    return distance_in_km
```

### 3. Model Training
- **Algorithm:** XGBoost Regressor
- **Parameters:**
  - n_estimators: 100 trees
  - learning_rate: 0.1
  - max_depth: 6
- **Train/Test Split:** 80/20

### 4. Prediction
Input features → Model → Predicted delivery time (minutes)

---

## 🎯 Dataset

### Required Columns
- `Delivery_person_Age` - Age of delivery partner
- `Delivery_person_Ratings` - Rating (1-5)
- `Restaurant_latitude` & `Restaurant_longitude`
- `Delivery_location_latitude` & `Delivery_location_longitude`
- `Time_Orderd` - Order time
- `Order_Date` - Order date
- `Weatherconditions` - Current weather
- `Road_traffic_density` - Traffic level
- `Type_of_order` - Snack/Meal/Drinks/Buffet
- `Type_of_vehicle` - Delivery vehicle type
- `Festival` - Festival day indicator
- `City` - City type
- `multiple_deliveries` - Number of concurrent orders
- `Time_taken(min)` - **Target variable**

### Data Format
```csv
ID,Delivery_person_Age,Delivery_person_Ratings,Restaurant_latitude,...
0x4607,28,4.5,19.0760,72.8777,...,25
```

---

## 🛠️ Customization

### Adjust Model Parameters

Edit `delivery_predictor.py`:

```python
model = xgb.XGBRegressor(
    n_estimators=150,      # Try 150 trees
    learning_rate=0.05,    # Slower learning
    max_depth=8,           # Deeper trees
    random_state=42,
    n_jobs=-1
)
```

### Add More Features

```python
# Add day of week
df['day_of_week'] = df['Order_Date'].dt.dayofweek

# Add preparation time estimate
df['prep_time_estimate'] = df['Type_of_order'].map({
    'Snack': 10,
    'Meal': 20,
    'Drinks': 5,
    'Buffet': 30
})
```

### Customize UI Theme

Edit `app.py` CSS:

```python
st.markdown("""
    <style>
    .main-header {
        color: #YOUR_COLOR;  # Change header color
    }
    </style>
""", unsafe_allow_html=True)
```

---

## 🐛 Troubleshooting

### Issue: Module not found
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Model file not found
```bash
python delivery_predictor.py  # Train model first
```

### Issue: Streamlit won't start
```bash
streamlit run app.py --server.port 8502  # Try different port
```

### Issue: Poor predictions
- Check if you have enough training data (>1000 rows recommended)
- Verify data quality (no extreme outliers)
- Retrain model with more data

---

## 🚀 Deployment

### Deploy to Streamlit Cloud (Free!)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Click "Deploy"
5. Share your app URL! 🎉

### Deploy to Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a new branch (`git checkout -b feature/improvement`)
3. **Make** your changes
4. **Commit** (`git commit -am 'Add new feature'`)
5. **Push** (`git push origin feature/improvement`)
6. **Create** a Pull Request

### Ideas for Contributions
- 🗺️ Add Google Maps integration
- 📊 Create analytics dashboard
- 🔔 Add notification system
- 🌍 Multi-language support
- 📱 Mobile app version

---


<div align="center">



---

**Happy Coding! 🚀**
