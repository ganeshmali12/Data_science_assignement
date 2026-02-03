"""
Titanic Survival Prediction App
Logistic Regression Model Deployment using Streamlit
"""

import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_model():
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Title and description
st.title("🚢 Titanic Survival Predictor")
st.markdown("Predict whether a passenger would survive the Titanic disaster based on their characteristics.")
st.markdown("---")

# Input form
st.subheader("Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: f"{x}st Class" if x == 1 else (f"{x}nd Class" if x == 2 else f"{x}rd Class")
    )

    sex = st.selectbox(
        "Sex",
        options=["Female", "Male"]
    )

    age = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=30
    )

    sibsp = st.number_input(
        "Siblings/Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0
    )

with col2:
    parch = st.number_input(
        "Parents/Children Aboard",
        min_value=0,
        max_value=6,
        value=0
    )

    fare = st.slider(
        "Fare (in pounds)",
        min_value=0.0,
        max_value=520.0,
        value=50.0,
        step=0.5
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["Southampton", "Cherbourg", "Queenstown"]
    )

st.markdown("---")

# Encode inputs
sex_encoded = 1 if sex == "Male" else 0
embarked_mapping = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
embarked_encoded = embarked_mapping[embarked]

# Create feature array
# Features: ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict button
if st.button("Predict Survival", type="primary", use_container_width=True):
    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ **Survived!**")
        st.balloons()
    else:
        st.error(f"❌ **Did Not Survive**")

    # Show probabilities
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Survival Probability", f"{probability[1]*100:.1f}%")
    with col2:
        st.metric("Non-Survival Probability", f"{probability[0]*100:.1f}%")

    # Progress bar for survival probability
    st.progress(probability[1])

# Sidebar with info
st.sidebar.header("About")
st.sidebar.info(
    """
    This app uses a **Logistic Regression** model trained on the Titanic dataset
    to predict passenger survival.

    **Features used:**
    - Passenger Class
    - Sex
    - Age
    - Siblings/Spouses Aboard
    - Parents/Children Aboard
    - Fare
    - Port of Embarkation
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance:**")
st.sidebar.text("Accuracy: 65.36%")
st.sidebar.text("ROC-AUC: 0.5216")

st.sidebar.markdown("---")
st.sidebar.caption("Assignment 7: Logistic Regression")
