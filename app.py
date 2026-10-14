import streamlit as st
import pandas as pd
import pickle

# Path to your image file or URL
BANNER_IMAGE_PATH = "banner.png"

st.image(r"C:\Users\Sanju\Downloads\ChatGPT Image Oct 7, 2026, 03_05_16 PM.png", use_container_width=True)

# Load Trained Pipeline Model
@st.cache_resource
def load_model():
    with open('swiggy.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("Machine Learning-Powered Swiggy Delivery Time Predictor")
st.divider()

# Input Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    # Column 1: Order & Location Details
    with col1:
        st.subheader("Location & Distance")
        distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=3.5, step=0.1)
        city_type = st.selectbox("City Type", ["Urban", "Metropolitian", "Semi-Urban"])
        pickup_time_minutes = st.number_input("Pickup Time (mins)", min_value=1.0, max_value=60.0, value=10.0, step=1.0)
        multiple_deliveries = st.selectbox("Multiple Deliveries", [0.0, 1.0, 2.0, 3.0])

    # Column 2: Delivery Partner & Vehicle Info
    with col2:
        st.subheader("Rider & Vehicle")
        age = st.number_input("Delivery Person Age", min_value=18, max_value=65, value=28)
        ratings = st.slider("Delivery Person Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        type_of_vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
        vehicle_condition = st.selectbox("Vehicle Condition Rating", [0, 1, 2])

    # Column 3: Weather & Environment Details
    with col3:
        st.subheader("Weather & Traffic")
        weather = st.selectbox("Weather Condition", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"])
        traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
        type_of_order = st.selectbox("Type of Order", ["Snacks", "Meal", "Drinks", "Buffet"])
        festival = st.selectbox("Festival", ["No", "Yes"])

    st.subheader("Date & Time Context")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        order_day = st.number_input("Order Day", min_value=1, max_value=31, value=15)
    with t2:
        order_month = st.number_input("Order Month", min_value=1, max_value=12, value=3)
    with t3:
        order_day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with t4:
        order_time_hour = st.number_input("Order Hour (0-23)", min_value=0, max_value=23, value=14)
    with t5:
        order_time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    # Derive weekend flag
    is_weekend = "1" if order_day_of_week.lower() in ["saturday", "sunday"] else "0"

    submit_button = st.form_submit_button("Predict Delivery Time", use_container_width=True)

# Prediction Logic
if submit_button:
    # Match exact feature names and cleaned string formats expected by preprocessor
    input_data = pd.DataFrame({
        'age': [float(age)],
        'ratings': [float(ratings)],
        'weather': [weather.strip().lower()],
        'traffic': [traffic.strip().lower()],
        'vehicle_condition': [int(vehicle_condition)],
        'type_of_order': [type_of_order.strip()],
        'type_of_vehicle': [type_of_vehicle.strip().lower()],
        'multiple_deliveries': [float(multiple_deliveries)],
        'festival': [festival.strip().lower()],
        'city_type': [city_type.strip().lower()],
        'order_day': [int(order_day)],
        'order_month': [int(order_month)],
        'order_day_of_week': [order_day_of_week.strip().lower()],
        'is_weekend': [is_weekend],
        'pickup_time_minutes': [float(pickup_time_minutes)],
        'order_time_hour': [float(order_time_hour)],
        'order_time_of_day': [order_time_of_day.strip().lower()],
        'distance': [float(distance)]
    })

    try:
        prediction = model.predict(input_data)[0]
        predicted_time = f"{float(prediction):.2f}"

        # Large font display for predicted delivery time
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-top: 10px; margin-bottom: 20px;">
                <h3 style="color: #333333; margin: 0;">Estimated Delivery Time</h3>
                <h1 style="color: #ff6600; font-size: 56px; font-weight: bold; margin: 10px 0;">{predicted_time} Mins</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Distance context explanation
        st.info(
            f"Distance: **{distance} km** | For a distance of {distance} km under {weather.lower()} weather and {traffic.lower()} traffic conditions, the predicted delivery time is estimated at {predicted_time} minutes."
        )

    except Exception as e:
        st.error(f"Error generating prediction: {e}")


