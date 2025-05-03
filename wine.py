import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="wine_recommendation.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Wine Quality Prediction")
st.write("Masukkan parameter kimia dari wine untuk memprediksi kualitasnya.")

# Input pengguna berdasarkan fitur dari dataset
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.5, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, 2.5)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.05)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 46.0)
density = st.number_input("Density", 0.9900, 1.0050, 0.9968)
pH = st.number_input("pH", 2.5, 4.5, 3.3)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.65)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 10.0)
quality = st.number_input("Quality", 0.0, 10.0, 5.0)

# Tombol prediksi
if st.button("Prediksi Kualitas Wine"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol, quality]])
    
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = np.argmax(prediction)
    quality = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Prediksi kualitas wine: **{quality}**")
