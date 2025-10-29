import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import plotly.graph_objects as go

# =====================================
# Load Model and Encoders
# =====================================
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('model.h5')

    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)

    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# =====================================
# Page Config
# =====================================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# =====================================
# Custom CSS
# =====================================
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f0f9ff 0%, #cbebff 50%, #e0f2fe 100%);
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        .main-title {
            text-align: center;
            color: #1e3a8a;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            letter-spacing: 1px;
        }
        .subtitle {
            text-align: center;
            color: #334155;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .input-card {
            background-color: #ffffffd9;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-top: 1rem;
        }
        .predict-btn button {
            background: linear-gradient(90deg, #3b82f6, #2563eb);
            color: white !important;
            font-weight: 600 !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 8px !important;
            transition: all 0.3s ease-in-out !important;
        }
        .predict-btn button:hover {
            background: linear-gradient(90deg, #2563eb, #1e40af);
            transform: scale(1.05);
        }
        .result-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            text-align: center;
            margin-top: 2rem;
        }
        .stay {
            color: #16a34a;
            font-size: 1.3rem;
            font-weight: 700;
        }
        .churn {
            color: #dc2626;
            font-size: 1.3rem;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================
# Title Section
# =====================================
st.markdown("<div class='main-title'>Bank Customer Churn Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a customer will stay or leave using a deep learning model</div>", unsafe_allow_html=True)

# =====================================
# Input Section
# =====================================
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    num_of_products = st.slider("Number of Products", 1, 4, 2)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    balance = st.number_input("Balance", min_value=0.0, step=100.0, value=70000.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0, value=50000.0)
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1], index=1)
    is_active_member = st.selectbox("Is Active Member?", [0, 1], index=1)
st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# Prediction Button
# =====================================
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
predict_clicked = st.button("Predict Churn", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# Prediction Logic
# =====================================
if predict_clicked:
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    churn_prob = float(prediction[0][0])

    if churn_prob > 0.5:
        churn_text = "Customer likely to CHURN"
        result_class = "churn"
    else:
        churn_text = "Customer likely to STAY"
        result_class = "stay"

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='{result_class}'>{churn_text}</div>", unsafe_allow_html=True)
    st.metric(label="Churn Probability", value=f"{churn_prob:.2f}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 40], 'color': "#bbf7d0"},
                {'range': [40, 70], 'color': "#fde68a"},
                {'range': [70, 100], 'color': "#fecaca"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("View Processed Input Data"):
        st.dataframe(input_data)
