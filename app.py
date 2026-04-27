import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    .prediction-card {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-style {
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_models():
    model = joblib.load("student_performance_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return model, scaler, pca

try:
    model, scaler, pca = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure .pkl files are in the same directory.")
    st.stop()

# Sidebar for information
st.sidebar.title("About the Project")
st.sidebar.info("""
    This app uses a Linear Regression model to predict a student's **Performance Index** based on academic and lifestyle factors.
    
    **Features used:**
    - Hours Studied
    - Previous Scores
    - Extracurricular Activities
    - Sleep Hours
    - Sample Papers Practiced
""")

# Main Content
st.markdown("<h1 class='header-style'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Student Details")
    
    with st.form("student_form"):
        hours_studied = st.number_input("Hours Studied", min_value=1, max_value=9, value=5, help="Number of hours spent studying daily.")
        previous_scores = st.number_input("Previous Scores", min_value=40, max_value=99, value=70, help="Scores obtained in previous tests (40-99).")
        extracurricular = st.selectbox("Extracurricular Activities", options=["No", "Yes"], index=0)
        sleep_hours = st.number_input("Sleep Hours", min_value=4, max_value=9, value=7, help="Average daily sleep duration.")
        sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=9, value=5)
        
        submit_button = st.form_submit_button(label="Predict Performance")

if submit_button:
    # Prepare data for prediction
    # Extracurricular: Yes -> 1, No -> 0
    extracurricular_numeric = 1 if extracurricular == "Yes" else 0
    
    input_data = pd.DataFrame({
        "Hours Studied": [hours_studied],
        "Previous Scores": [previous_scores],
        "Extracurricular Activities": [extracurricular_numeric],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers]
    })
    
    # Preprocessing
    try:
        # Scale
        scaled_data = scaler.transform(input_data)
        # PCA
        pca_data = pca.transform(scaled_data)
        # Predict
        prediction = model.predict(pca_data)[0]
        
        with col2:
            st.subheader("🎯 Prediction Result")
            
            # Display Prediction in a card
            st.markdown(f"""
                <div class='prediction-card'>
                    <h3>Estimated Performance Index</h3>
                    <h1 style='color: #4CAF50; font-size: 60px;'>{prediction:.2f}</h1>
                    <p>Based on the provided metrics, this student is likely to achieve this score.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional Insights
            st.markdown("---")
            st.write("### 💡 Improvement Tips")
            if prediction < 60:
                st.warning("Score is on the lower side. Suggest increasing **Hours Studied** and practicing more **Sample Papers**.")
            elif prediction < 85:
                st.info("Good performance! Balanced sleep and study routines are key to maintaining this index.")
            else:
                st.success("Excellent prediction! Keep up the great work.")
                
            # Radar chart style comparison (simple bar for now)
            st.write("### 📊 Metrics Comparison")
            metrics = {
                "Hours Studied": hours_studied / 9,
                "Previous Scores": previous_scores / 100,
                "Sleep Hours": sleep_hours / 9,
                "Sample Papers": sample_papers / 9
            }
            st.bar_chart(pd.Series(metrics))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    with col2:
        st.info("👈 Enter details on the left and click 'Predict Performance' to see the results.")
        # Placeholder image/icon
        st.image("https://img.freepik.com/free-vector/study-concept-illustration_114360-1015.jpg", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with Streamlit & Scikit-Learn | Developed by Biswajeet Kumar</p>", unsafe_allow_html=True)
