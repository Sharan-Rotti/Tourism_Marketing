# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import accuracy_score

# # -----------------------------
# # Page Configuration
# # -----------------------------
# st.set_page_config(
#     page_title="Tourism Marketing – ML App",
#     page_icon="🌍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # -----------------------------
# # Custom CSS for Professional UI
# # -----------------------------
# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #0e1117;
#     }
#     .block-container {
#         padding-top: 2rem;
#     }
#     h1, h2, h3 {
#         color: #eaeaea;
#     }
#     .stButton>button {
#         border-radius: 8px;
#         height: 3em;
#         width: 100%;
#         font-size: 16px;
#         font-weight: 600;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # -----------------------------
# # Load Model
# # -----------------------------
# @st.cache_resource
# def load_model():
#     model = joblib.load("random_forest_model.joblib")
#     return model

# model = load_model()

# # -----------------------------
# # Sidebar
# # -----------------------------
# with st.sidebar:
#     st.title("🌍 Tourism Marketing ML")
#     st.markdown("---")
#     st.markdown(
#         """
#         **Model:** Random Forest Classifier  
#         **Use Case:** Customer Travel Purchase Prediction  
#         **Built by:** ML Engineer
#         """
#     )
#     st.markdown("---")
#     st.info("Fill the inputs on the right panel and click **Predict**")

# # -----------------------------
# # Main Title
# # -----------------------------
# st.title("🎯 Tourism Marketing Prediction System")
# st.markdown(
#     "Predict whether a customer is **likely to purchase a travel package** based on demographic and behavioral data."
# )

# st.markdown("---")

# # -----------------------------
# # Input Section
# # -----------------------------
# st.subheader("🧾 Customer Information")

# col1, col2, col3 = st.columns(3)

# with col1:
#     age = st.slider("Age", 18, 70, 30)
#     annual_income = st.number_input("Annual Income", min_value=10000, max_value=500000, step=5000)

# with col2:
#     family_size = st.slider("Family Size", 1, 10, 3)
#     work_experience = st.slider("Work Experience (Years)", 0, 40, 5)

# with col3:
#     graduate = st.selectbox("Graduate", ["Yes", "No"])
#     frequent_flyer = st.selectbox("Frequent Flyer", ["Yes", "No"])

# # Encode categorical values (MUST MATCH TRAINING LOGIC)
# graduate = 1 if graduate == "Yes" else 0
# frequent_flyer = 1 if frequent_flyer == "Yes" else 0

# # -----------------------------
# # Prediction
# # -----------------------------
# st.markdown("---")

# predict_btn = st.button("🔮 Predict Purchase")

# if predict_btn:
#     input_data = np.array([[age, annual_income, family_size, work_experience, graduate, frequent_flyer]])

#     prediction = model.predict(input_data)[0]
#     prediction_proba = model.predict_proba(input_data)

#     st.markdown("---")
#     st.subheader("📊 Prediction Result")

#     if prediction == 1:
#         st.success("✅ Customer is **LIKELY to purchase** the travel package")
#     else:
#         st.error("❌ Customer is **NOT likely to purchase** the travel package")

#     st.markdown("#### Confidence Score")
#     st.progress(float(np.max(prediction_proba)))

# # -----------------------------
# # Footer
# # -----------------------------
# st.markdown("---")
# st.markdown(
#     "<center><small>Built with ❤️ using Streamlit & Scikit-learn</small></center>",
#     unsafe_allow_html=True
# )


# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # =============================
# # Page Configuration
# # =============================
# st.set_page_config(
#     page_title="Tourism Marketing Prediction",
#     page_icon="🌍",
#     layout="wide"
# )

# # =============================
# # Load Model
# # =============================
# @st.cache_resource
# def load_model():
#     return joblib.load("random_forest_model.joblib")

# model = load_model()

# # =============================
# # Sidebar
# # =============================
# st.sidebar.title("🌍 Tourism Marketing ML App")
# st.sidebar.markdown("---")
# st.sidebar.info(
#     "Predict whether a customer will purchase a travel package (ProdTaken)."
# )
# st.sidebar.markdown("**Model:** Random Forest Classifier")
# st.sidebar.markdown(f"**Expected Features:** {model.n_features_in_}")

# # =============================
# # Main Title
# # =============================
# st.title("🎯 Tourism Package Purchase Prediction")
# st.markdown("Fill all customer details below and click **Predict**")
# st.markdown("---")

# # =============================
# # Input Sections
# # =============================
# col1, col2, col3 = st.columns(3)

# with col1:
#     age = st.slider("Age", 18, 70, 30)
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     city_tier = st.selectbox("City Tier", [1, 2, 3])
#     occupation = st.selectbox(
#         "Occupation",
#         ["Salaried", "Small Business", "Large Business", "Free Lancer"]
#     )
#     monthly_income = st.number_input("Monthly Income", 1000, 100000, 25000)

# with col2:
#     typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
#     duration_pitch = st.slider("Duration Of Pitch (mins)", 0, 60, 15)
#     followups = st.slider("Number Of Followups", 0, 10, 2)
#     product_pitched = st.selectbox(
#         "Product Pitched",
#         ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
#     )
#     property_star = st.selectbox("Preferred Property Star", [3, 4, 5])

# with col3:
#     persons_visiting = st.slider("Number Of Person Visiting", 1, 10, 2)
#     marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
#     trips = st.slider("Number Of Trips", 0, 20, 2)
#     passport = st.selectbox("Passport", [0, 1])
#     pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
#     own_car = st.selectbox("Own Car", [0, 1])
#     children_visiting = st.slider("Number Of Children Visiting", 0, 5, 0)
#     designation = st.selectbox(
#         "Designation",
#         ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
#     )

# # =============================
# # Create Input DataFrame (MATCH TRAINING FEATURES EXACTLY)
# # =============================
# input_df = pd.DataFrame([{
#     "Age": age,
#     "TypeofContact": typeof_contact,
#     "CityTier": city_tier,
#     "DurationOfPitch": duration_pitch,
#     "Occupation": occupation,
#     "Gender": gender,
#     "NumberOfPersonVisiting": persons_visiting,
#     "NumberOfFollowups": followups,
#     "ProductPitched": product_pitched,
#     "PreferredPropertyStar": property_star,
#     "MaritalStatus": marital_status,
#     "NumberOfTrips": trips,
#     "Passport": passport,
#     "PitchSatisfactionScore": pitch_score,
#     "OwnCar": own_car,
#     "NumberOfChildrenVisiting": children_visiting,
#     "Designation": designation,
#     "MonthlyIncome": monthly_income
# }])

# # =============================
# # Prediction
# # =============================
# st.markdown("---")

# if st.button("🔮 Predict"):
#     prediction = model.predict(input_df)[0]
#     proba = model.predict_proba(input_df)

#     st.subheader("📊 Prediction Result")

#     if prediction == 1:
#         st.success("✅ Customer is LIKELY to purchase the travel package")
#     else:
#         st.error("❌ Customer is NOT likely to purchase the travel package")

#     st.markdown("### Confidence")
#     st.progress(float(np.max(proba)))

# # =============================
# # Footer
# # =============================
# st.markdown("---")
# st.caption("Built with Streamlit & Scikit-learn | ML Engineer Project")




import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Tourism Marketing Prediction",
    page_icon="🌍",
    layout="wide"
)

# =============================
# Load Model (SAFE + DEBUG FRIENDLY)
# =============================
@st.cache_resource

def load_model():
    model_path = "random_forest_model.joblib"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Ensure 'random_forest_model.joblib' is in the project root.")
        st.stop()

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error("❌ Failed to load model. This usually happens due to Python/sklearn version mismatch or corrupted file.")
        st.exception(e)
        st.stop()

model = load_model()

# =============================
# Sidebar
# =============================
st.sidebar.title("🌍 Tourism Marketing ML App")
st.sidebar.markdown("---")
st.sidebar.info("Predict whether a customer will purchase a travel package")
st.sidebar.markdown("**Model:** Random Forest Classifier")
st.sidebar.markdown(f"**Expected Features:** {model.n_features_in_}")

# =============================
# Main Title
# =============================
st.title("🎯 Tourism Package Purchase Prediction")
st.markdown("Fill all customer details below and click **Predict**")
st.markdown("---")

# =============================
# Input Sections
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=25000)

with col2:
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    duration_pitch = st.slider("Duration Of Pitch (mins)", 0, 60, 15)
    followups = st.slider("Number Of Followups", 0, 10, 2)
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    property_star = st.selectbox("Preferred Property Star", [3, 4, 5])

with col3:
    persons_visiting = st.slider("Number Of Person Visiting", 1, 10, 2)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    trips = st.slider("Number Of Trips", 0, 20, 2)
    passport = st.selectbox("Passport", [0, 1])
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Own Car", [0, 1])
    children_visiting = st.slider("Number Of Children Visiting", 0, 5, 0)
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

# =============================
# Create Input DataFrame (MATCH TRAINING SCHEMA)
# =============================
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons_visiting,
    "NumberOfFollowups": followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# =============================
# Prediction
# =============================
st.markdown("---")

if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("✅ Customer is LIKELY to purchase the travel package")
        else:
            st.error("❌ Customer is NOT likely to purchase the travel package")

        st.markdown("### Confidence")
        st.progress(float(np.max(proba)))

    except Exception as e:
        st.error("Prediction failed due to input/schema mismatch")
        st.exception(e)

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Built with Streamlit & Scikit-learn | ML Engineer Project")
