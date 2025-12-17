import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Tourism Marketing – ML App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Professional UI
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #eaeaea;
    }
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.joblib")
    return model

model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🌍 Tourism Marketing ML")
    st.markdown("---")
    st.markdown(
        """
        **Model:** Random Forest Classifier  
        **Use Case:** Customer Travel Purchase Prediction  
        **Built by:** ML Engineer
        """
    )
    st.markdown("---")
    st.info("Fill the inputs on the right panel and click **Predict**")

# -----------------------------
# Main Title
# -----------------------------
st.title("🎯 Tourism Marketing Prediction System")
st.markdown(
    "Predict whether a customer is **likely to purchase a travel package** based on demographic and behavioral data."
)

st.markdown("---")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 70, 30)
    annual_income = st.number_input("Annual Income", min_value=10000, max_value=500000, step=5000)

with col2:
    family_size = st.slider("Family Size", 1, 10, 3)
    work_experience = st.slider("Work Experience (Years)", 0, 40, 5)

with col3:
    graduate = st.selectbox("Graduate", ["Yes", "No"])
    frequent_flyer = st.selectbox("Frequent Flyer", ["Yes", "No"])

# Encode categorical values (MUST MATCH TRAINING LOGIC)
graduate = 1 if graduate == "Yes" else 0
frequent_flyer = 1 if frequent_flyer == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------
st.markdown("---")

predict_btn = st.button("🔮 Predict Purchase")

if predict_btn:
    input_data = np.array([[age, annual_income, family_size, work_experience, graduate, frequent_flyer]])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Customer is **LIKELY to purchase** the travel package")
    else:
        st.error("❌ Customer is **NOT likely to purchase** the travel package")

    st.markdown("#### Confidence Score")
    st.progress(float(np.max(prediction_proba)))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit & Scikit-learn</small></center>",
    unsafe_allow_html=True
)

