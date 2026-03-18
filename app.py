import streamlit as st
import joblib
import numpy as np

# Load model and scaler (must be retrained without FWI)
model = joblib.load("forest_fire_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Forest Fire Prediction")

st.title("🔥 Forest Fire Prediction System")
st.write("Enter environmental conditions to predict fire risk")

# User input sliders (FWI removed)
Temperature = st.slider("Temperature (°C)", 22, 42, 30)
RH = st.slider("Relative Humidity (%)", 21, 90, 50)
FFMC = st.slider("FFMC", 28.6, 96.0, 80.0)
DMC = st.slider("DMC", 0.7, 65.9, 15.0)
ISI = st.slider("ISI", 0.0, 19.0, 5.0)

# Prediction button
if st.button("Predict Fire Risk"):

    # Data array without FWI
    data = np.array([[Temperature, RH, FFMC, DMC, ISI]])

    # Scale the data
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)
    fire_probability = probability[0][1] * 100

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"🔥 Fire Risk ({fire_probability:.2f}%)")
    else:
        st.success(f"✅ No Fire ({100-fire_probability:.2f}%)")

    st.progress(int(fire_probability))