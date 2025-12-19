import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.predict import predict_yield, load_best_model
from src.utils.data_info import MAPPED_CROPS, SEASONS, SOIL_TYPES

# Page configuration
st.set_page_config(
    page_title="AgriYield Predictor | Indra Kumar",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .card {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content {
        background-color: #1a237e;
    }
    h1, h2, h3 {
        color: #1b3022;
    }
    .stAlert {
        border-radius: 10px;
    }
    /* Brand footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #555;
        text-align: center;
        padding: 0.5rem;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1147/1147805.png", width=80)
    st.title("AgriYield Pro")
    st.markdown("---")
    page = st.radio("Navigation", ["Overview", "Yield Prediction", "Model Insights", "About"])
    
    st.markdown("---")
    st.info("**ML Engineering & Deployment** by Indra Kumar")
    st.caption("Developed for **Infosys Springboard AI Project**")
    st.markdown("Built with ❤️ for sustainable agriculture.")

# Branding Overlay
st.markdown("<div class='footer'>Built by Indra Kumar | Infosys Springboard AI Project © 2025</div>", unsafe_allow_html=True)

if page == "Overview":
    st.title("🌾 AgriYield Predictor Overview")
    st.markdown("### Empowering Agriculture through AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        The **AgriYield Predictor** is a production-grade machine learning system designed to provide 
        accurate crop yield forecasts based on environmental and soil conditions. 
        
        #### Key Features:
        - **Precision Forecasting**: Leverages advanced regression models trained on historical data.
        - **Data-Driven Insights**: Considers Nitrogen, Phosphorus, Potassium levels, and weather patterns.
        - **User-Centric Design**: Simple interface for instant predictions.
        
        #### How it works:
        1. Input your current soil nutrient levels.
        2. Provide environmental data (Temperature, Humidity, Rainfall).
        3. Select your Crop, Season, and Soil type.
        4. Get an instant yield prediction in tonnes per hectare (t/ha).
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=600", caption="Smart Farming with AI")

elif page == "Yield Prediction":
    st.title("🔮 Instant Yield Prediction")
    st.markdown("Provide the details below to estimate your crop yield.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🧪 Soil Nutrients")
            n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
            p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
            k = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
            ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5)
            
        with col2:
            st.markdown("### 🌡️ Environment")
            temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
            hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=3000.0, value=500.0)
            
        with col3:
            st.markdown("### 🚜 Crop Info")
            crop = st.selectbox("Crop Type", options=sorted(MAPPED_CROPS))
            season = st.selectbox("Season", options=sorted(SEASONS))
            soil = st.selectbox("Soil Type", options=sorted(SOIL_TYPES))
            
        st.markdown("---")
        submit = st.form_submit_button("Predict Yield Now")
        
    if submit:
        input_data = {
            'N': n, 'P': p, 'K': k,
            'temperature': temp, 'humidity': hum,
            'ph': ph, 'rainfall': rain,
            'Crop': crop, 'Season': season, 'Soil_Type': soil
        }
        
        with st.spinner("Processing data and running inference..."):
            try:
                prediction = predict_yield(input_data)
                st.balloons()
                st.success("Analysis Complete!")
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='margin:0; color:#555;'>Estimated Yield</h4>
                            <h1 style='margin:0; color:#2e7d32;'>{prediction:.3f}</h1>
                            <p style='margin:0; color:#777;'>tonnes per hectare (t/ha)</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with res_col2:
                    st.info(f"The model predicted a yield of **{prediction:.3f} t/ha** for **{crop}** in **{season}** season with **{soil}** soil.")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

elif page == "Model Insights":
    st.title("📊 Model Insights & Performance")
    
    try:
        metrics_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'model_metrics.csv')
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            st.write("### Model Performance Comparison")
            st.dataframe(metrics_df, use_container_width=True)
            
            st.bar_chart(metrics_df.set_index('model')['r2'])
        else:
            st.warning("Model metrics not found. Please run training first.")
            
        st.markdown("""
        #### Why these metrics matter?
        - **R² Score**: Indicates how well the model explains the variability of the target. Higher is better.
        - **RMSE**: Root Mean Squared Error. Represents the average deviation from actual values. Lower is better.
        - **MAE**: Mean Absolute Error. The average absolute difference between predicted and actual yield.
        """)
    except Exception as e:
        st.error(f"Error loading insights: {e}")

elif page == "About":
    st.title("👨‍💻 About This Project")
    st.markdown(f"""
    ### ML Engineering Improvements by Indra Kumar
    
    This project was developed as a flagship submission for the **Infosys Springboard AI Project**. 
    It was restructured from a notebook-centric exploratory project into a production-grade machine learning application.
    
    **Architectural Improvements:**
    - **Modular Codebase**: Split logic into `data`, `features`, `models`, and `training` modules.
    - **Production Inference**: Robust model loading and prediction pipeline.
    - **Modern UI**: Completely redesigned Streamlit interface with improved UX.
    - **Clean Data Flow**: Structured directory layout for raw/processed data and model artifacts.
    
    **Developer:** Indra Kumar
    **Role:** ML Engineer
    """)
    
    st.markdown("---")
    st.markdown("#### Preservation of Attribution")
    st.markdown("Original research and dataset gathering by the initial project authors.")
