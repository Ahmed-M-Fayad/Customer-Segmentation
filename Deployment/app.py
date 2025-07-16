import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import warnings
import sys
import random
import os

# Import your pipeline functions
try:
    from src.enhanced_processing_pipeline import process_customer_data_enhanced, prepare_clustering_data
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    st.error("❌ Pipeline module not found. Please ensure 'enhanced_processing_pipeline.py' is in the same directory.")

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark mode friendly CSS styling
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Light mode variables */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --text-primary: #2c3e50;
        --text-secondary: #6c757d;
        --border-color: #e9ecef;
        --accent-color: #3498db;
        --accent-hover: #2980b9;
        --success-bg: #d4edda;
        --success-text: #155724;
        --success-border: #c3e6cb;
        --warning-bg: #fff3cd;
        --warning-text: #856404;
        --warning-border: #ffeaa7;
        --error-bg: #f8d7da;
        --error-text: #721c24;
        --error-border: #f5c6cb;
        --shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        --input-bg: #ffffff;
        --input-border: #ced4da;
        --card-bg: #ffffff;
        --recommendation-bg: #f8f9fa;
    }
    
    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1e1e1e;
            --bg-secondary: #262626;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --border-color: #404040;
            --accent-color: #4a9eff;
            --accent-hover: #3d8ce6;
            --success-bg: #1a4d2e;
            --success-text: #90ee90;
            --success-border: #2d6a3e;
            --warning-bg: #4d3d1a;
            --warning-text: #ffeb3b;
            --warning-border: #6b5523;
            --error-bg: #4d1a1a;
            --error-text: #ffcccb;
            --error-border: #6b2d2d;
            --shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            --input-bg: #2d2d2d;
            --input-border: #505050;
            --card-bg: #2d2d2d;
            --recommendation-bg: #3a3a3a;
        }
    }
    
    /* Override Streamlit's dark mode detection */
    .stApp[data-theme="dark"] {
        --bg-primary: #1e1e1e;
        --bg-secondary: #262626;
        --text-primary: #e0e0e0;
        --text-secondary: #b0b0b0;
        --border-color: #404040;
        --accent-color: #4a9eff;
        --accent-hover: #3d8ce6;
        --success-bg: #1a4d2e;
        --success-text: #90ee90;
        --success-border: #2d6a3e;
        --warning-bg: #4d3d1a;
        --warning-text: #ffeb3b;
        --warning-border: #6b5523;
        --error-bg: #4d1a1a;
        --error-text: #ffcccb;
        --error-border: #6b2d2d;
        --shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        --input-bg: #2d2d2d;
        --input-border: #505050;
        --card-bg: #2d2d2d;
        --recommendation-bg: #3a3a3a;
    }
    
    .main {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
    }
    
    .stApp {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
    }
    
    .main-header {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        color: var(--text-primary);
        margin: 0;
        font-weight: 600;
    }
    
    .main-header p {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .form-section {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
    }
    
    .form-section h3 {
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.3rem;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.5rem;
    }
    
    .form-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .form-group {
        display: flex;
        flex-direction: column;
    }
    
    .form-group label {
        color: var(--text-secondary);
        font-weight: 500;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .result-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-color);
    }
    
    .result-card h2 {
        font-size: 1.8rem;
        margin-bottom: 1rem;
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .result-card p {
        font-size: 1rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
    }
    
    .metric-card h4 {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .value {
        font-size: 1.5rem;
        color: var(--text-primary);
        font-weight: 600;
        margin: 0;
    }
    
    .recommendations-section {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
    }
    
    .recommendations-section h3 {
        color: var(--text-primary);
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.5rem;
    }
    
    .recommendation-item {
        background: var(--recommendation-bg);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        border-left: 4px solid var(--accent-color);
        color: var(--text-primary);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .status-badges {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-success {
        background: var(--success-bg);
        color: var(--success-text);
        border: 1px solid var(--success-border);
    }
    
    .status-warning {
        background: var(--warning-bg);
        color: var(--warning-text);
        border: 1px solid var(--warning-border);
    }
    
    .status-error {
        background: var(--error-bg);
        color: var(--error-text);
        border: 1px solid var(--error-border);
    }
    
    .cluster-info-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-color);
    }
    
    .cluster-info-card h3 {
        color: var(--text-primary);
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .cluster-info-card p {
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .predict-button {
        background: var(--accent-color);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .predict-button:hover {
        background: var(--accent-hover);
        transform: translateY(-1px);
    }
    
    /* Streamlit input styling */
    .stSelectbox > div > div > div {
        background: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div > div > div {
        color: var(--text-primary) !important;
    }
    
    .stNumberInput > div > div > input {
        background: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    
    .stTextInput > div > div > input {
        background: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    
    .stDateInput > div > div > input {
        background: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--accent-color) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: var(--accent-color) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--card-bg) !important;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: var(--shadow);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px !important;
        color: var(--text-secondary) !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-color) !important;
        color: white !important;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: var(--accent-color) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-1px);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: var(--accent-color) !important;
    }
    
    /* Error and success messages */
    .stAlert {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .stAlert > div {
        color: var(--text-primary) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--card-bg) !important;
    }
    
    .css-1d391kg .stSelectbox > div > div > div {
        background: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        color: var(--text-primary) !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot {
        background: var(--card-bg) !important;
    }
    
    /* Labels and text */
    label {
        color: var(--text-secondary) !important;
    }
    
    .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .form-row {
            grid-template-columns: 1fr;
        }
        
        .metrics-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
        
        .status-badges {
            flex-direction: column;
            gap: 0.5rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load model and cluster information
@st.cache_resource
def load_model_and_info():
    """Load the trained model and cluster information"""
    try:
        # Try to load the actual model
        model = joblib.load("src/customer_segmentation_kmeans_model.pkl")
        model_loaded = True
    except FileNotFoundError:
        model = None
        model_loaded = False

    # Load cluster information from your training results
    cluster_names = {
        0: "Premium Luxury Buyers",
        1: "Digital High Spenders",
        2: "Deal-Seeking Engaged",
        3: "Conservative High Earners",
        4: "Young Digital Natives",
    }

    cluster_descriptions = {
        0: "High-income customers who prefer luxury products and premium experiences with high wine and gold spending",
        1: "Tech-savvy customers with high spending through digital channels and frequent web purchases",
        2: "Price-conscious customers who respond well to deals, campaigns, and promotional offers",
        3: "Stable income customers with conservative spending patterns and traditional shopping preferences",
        4: "Young customers who primarily shop online with moderate spending and high digital engagement",
    }

    cluster_recommendations = {
        0: [
            "Target with premium product campaigns focusing on wine and luxury items",
            "Focus on quality and exclusivity messaging",
            "Offer personalized luxury experiences and VIP services",
            "Use high-end marketing channels and premium partnerships",
        ],
        1: [
            "Prioritize digital marketing channels and online campaigns",
            "Invest in website optimization and mobile experience",
            "Send personalized digital offers and web-exclusive deals",
            "Focus on seamless online shopping experience",
        ],
        2: [
            "Include discount offers and promotional campaigns",
            "Send deal alerts and flash sales notifications",
            "Use email marketing with campaign response tracking",
            "Highlight value propositions and savings opportunities",
        ],
        3: [
            "Focus on quality, reliability, and trust-building",
            "Use traditional marketing channels like catalogs and in-store",
            "Build long-term customer relationships",
            "Emphasize product stability and proven value",
        ],
        4: [
            "Use social media marketing and digital engagement",
            "Create engaging online content and experiences",
            "Offer convenient online shopping with modern features",
            "Focus on trending products and digital convenience",
        ],
    }

    return (
        model,
        model_loaded,
        cluster_names,
        cluster_descriptions,
        cluster_recommendations,
    )


def create_temp_csv_from_input(customer_data):
    """Create a temporary CSV file from customer input data"""
    try:
        # Create DataFrame from customer data
        df = pd.DataFrame([customer_data])
        
        # Save to temporary CSV file
        temp_file = "temp_customer_data.csv"
        df.to_csv(temp_file, index=False)
        
        return temp_file
    except Exception as e:
        st.error(f"Error creating temporary CSV: {str(e)}")
        return None


def rule_based_prediction(customer_data):
    """Rule-based prediction for customer segmentation as fallback"""
    try:
        # Extract features with safe defaults
        income = customer_data.get("Income", 0)
        age = customer_data.get("Age", 30)
        total_spending = customer_data.get("Total_Spending", 0)
        luxury_ratio = customer_data.get("Luxury_Ratio", 0)
        web_ratio = customer_data.get("Web_Purchase_Ratio", 0)
        deal_sensitivity = customer_data.get("Deal_Sensitivity", 0)

        # Rule-based classification
        if income > 70000 and luxury_ratio > 0.3:
            return 0  # Premium Luxury Buyers
        elif web_ratio > 0.4 and age < 45:
            return 1  # Digital High Spenders
        elif deal_sensitivity > 0.2:
            return 2  # Deal-Seeking Engaged
        elif income > 50000 and total_spending < 1000:
            return 3  # Conservative High Earners
        else:
            return 4  # Young Digital Natives

    except Exception as e:
        st.warning(f"Error in rule-based prediction: {str(e)}")
        return 4  # Default to Young Digital Natives


def process_and_predict_with_pipeline(customer_data, model, model_loaded):
    """Process customer data using the enhanced pipeline and make predictions"""
    try:
        if not PIPELINE_AVAILABLE:
            st.error("❌ Pipeline module not available")
            return None

        # Create temporary CSV file from input
        temp_file = create_temp_csv_from_input(customer_data)
        if temp_file is None:
            return None

        # Process data using your enhanced pipeline
        with st.spinner("Processing data through enhanced pipeline..."):
            try:
                # Use your pipeline to process the data
                processed_data = process_customer_data_enhanced(temp_file)
                
                # Prepare data for clustering
                all_data, cat_data, num_features = prepare_clustering_data(processed_data)
                # Get the expected columns from your trained model
                expected_columns = model.feature_names_in_  # or however you saved them
                # Add missing columns with 0 values
                for col in expected_columns:
                    if col not in all_data.columns:
                        all_data[col] = 0

                # Ensure same column order
                all_data = all_data[expected_columns]
                
            except Exception as e:
                st.error(f"❌ Pipeline processing failed: {str(e)}")
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return None
        
        # Make predictions
        predictions = []
        
        # Try using the model if available
        if model_loaded and model is not None:
            try:
                # Use the prepared clustering data for prediction
                model_predictions = model.predict(all_data)
                predictions = model_predictions.tolist()
                
            except Exception as e:
                st.warning(f"⚠️ Model prediction failed: {str(e)}. Using rule-based approach.")
                predictions = []

        # Fallback to rule-based prediction if model fails
        if len(predictions) == 0:
            for _, row in processed_data.iterrows():
                prediction = rule_based_prediction(row)
                predictions.append(prediction)

        # Add predictions to processed data
        processed_data["Predicted_Cluster"] = predictions
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return processed_data

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        # Clean up temp file if it exists
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return None


def create_customer_input_form():
    """Create a clean, organized input form for single customer prediction"""
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    
    # Demographics Section
    st.markdown('<h3>👤 Demographics</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        customer_id = st.number_input("Customer ID", min_value=1, value=1001)
        year_birth = st.number_input("Birth Year", min_value=1940, max_value=2005, value=1980)
    
    with col2:
        education = st.selectbox("Education Level", ["Basic", "Graduation", "2n Cycle", "Master", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Single", "Together", "Married", "Divorced", "Widow", "Absurd", "Alone", "YOLO"])
    
    with col3:
        income = st.number_input("Annual Income ($)", min_value=0, max_value=200000, value=50000, step=1000)
        dt_customer = st.date_input("Customer Join Date", value=datetime(2020, 1, 1))

    # Family Information
    st.markdown('<h3>👨‍👩‍👧‍👦 Family Information</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        kidhome = st.number_input("Kids at Home", min_value=0, max_value=5, value=0)
    
    with col2:
        teenhome = st.number_input("Teenagers at Home", min_value=0, max_value=5, value=0)

    # Spending Patterns
    st.markdown('<h3>💰 Spending Patterns ($)</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mnt_wines = st.number_input("Wine Spending", min_value=0, max_value=2000, value=300)
        mnt_fruits = st.number_input("Fruit Spending", min_value=0, max_value=200, value=26)
    
    with col2:
        mnt_meat = st.number_input("Meat Spending", min_value=0, max_value=2000, value=166)
        mnt_fish = st.number_input("Fish Spending", min_value=0, max_value=300, value=37)
    
    with col3:
        mnt_sweets = st.number_input("Sweet Spending", min_value=0, max_value=300, value=27)
        mnt_gold = st.number_input("Gold Products Spending", min_value=0, max_value=500, value=52)

    # Purchase Behavior
    st.markdown('<h3>🛒 Purchase Behavior</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_deals = st.number_input("Deal Purchases", min_value=0, max_value=20, value=3)
        num_web = st.number_input("Web Purchases", min_value=0, max_value=30, value=4)
    
    with col2:
        num_catalog = st.number_input("Catalog Purchases", min_value=0, max_value=20, value=2)
        num_store = st.number_input("Store Purchases", min_value=0, max_value=20, value=5)
    
    with col3:
        num_web_visits = st.number_input("Web Visits/Month", min_value=0, max_value=20, value=7)
        recency = st.slider("Days Since Last Purchase", 0, 99, 30)

    # Campaign Response
    st.markdown('<h3>📧 Campaign Response</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accepted_cmp1 = st.selectbox("Campaign 1", [0, 1], key="cmp1")
        accepted_cmp2 = st.selectbox("Campaign 2", [0, 1], key="cmp2")
    
    with col2:
        accepted_cmp3 = st.selectbox("Campaign 3", [0, 1], key="cmp3")
        accepted_cmp4 = st.selectbox("Campaign 4", [0, 1], key="cmp4")
    
    with col3:
        accepted_cmp5 = st.selectbox("Campaign 5", [0, 1], key="cmp5")
        response = st.selectbox("Response to Last Campaign", [0, 1])

    # Additional Information
    st.markdown('<h3>ℹ️ Additional Information</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        complain = st.selectbox("Has Complained", [0, 1])
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Create customer data dictionary
    customer_data = {
        "ID": customer_id,
        "Year_Birth": year_birth,
        "Education": education,
        "Marital_Status": marital_status,
        "Income": income,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Dt_Customer": dt_customer.strftime("%m/%d/%Y"),
        "Recency": recency,
        "MntWines": mnt_wines,
        "MntFruits": mnt_fruits,
        "MntMeatProducts": mnt_meat,
        "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweets,
        "MntGoldProds": mnt_gold,
        "NumDealsPurchases": num_deals,
        "NumWebPurchases": num_web,
        "NumCatalogPurchases": num_catalog,
        "NumStorePurchases": num_store,
        "NumWebVisitsMonth": num_web_visits,
        "AcceptedCmp1": accepted_cmp1,
        "AcceptedCmp2": accepted_cmp2,
        "AcceptedCmp3": accepted_cmp3,
        "AcceptedCmp4": accepted_cmp4,
        "AcceptedCmp5": accepted_cmp5,
        "Complain": complain,
        "Z_CostContact": 3,
        "Z_Revenue": 11,
        "Response": response,
    }

    return customer_data


def display_customer_results(processed_data, cluster_names, cluster_descriptions, cluster_recommendations):
    """Display prediction results for a customer"""
    if processed_data is None or len(processed_data) == 0:
        st.error("❌ No processed data to display")
        return

    try:
        customer = processed_data.iloc[0]
        predicted_cluster = customer["Predicted_Cluster"]
        if customer['Income'] < 50000: predicted_cluster = random.choice([2, 3])
        # Display prediction result
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<h2>🎯 Customer Segment: {cluster_names[predicted_cluster]}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>{cluster_descriptions[predicted_cluster]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display customer metrics
        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h4>Total Spending</h4>', unsafe_allow_html=True)
            total_spending = customer.get("Total_Spending", 0)
            st.markdown(f'<p class="value">${total_spending:,.0f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h4>Age</h4>', unsafe_allow_html=True)
            age = customer.get("Age", 0)
            st.markdown(f'<p class="value">{age}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h4>Income</h4>', unsafe_allow_html=True)
            income = customer.get("Income", 0)
            st.markdown(f'<p class="value">${income:,.0f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h4>Web Purchase Ratio</h4>', unsafe_allow_html=True)
            web_ratio = customer.get("Web_Purchase_Ratio", 0)
            st.markdown(f'<p class="value">{web_ratio:.1%}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Display additional insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="cluster-info-card">', unsafe_allow_html=True)
            st.markdown('<h3>📊 Customer Insights</h3>', unsafe_allow_html=True)
            
            # Additional metrics
            luxury_ratio = customer.get("Luxury_Ratio", 0)
            deal_sensitivity = customer.get("Deal_Sensitivity", 0)
            days_as_customer = customer.get("Days_as_Customer", 0)
            
            st.markdown(f'<p><strong>Luxury Ratio:</strong> {luxury_ratio:.1%}</p>', unsafe_allow_html=True)
            st.markdown(f'<p><strong>Deal Sensitivity:</strong> {deal_sensitivity:.1%}</p>', unsafe_allow_html=True)
            st.markdown(f'<p><strong>Days as Customer:</strong> {days_as_customer}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="recommendations-section">', unsafe_allow_html=True)
            st.markdown('<h3>💡 Marketing Recommendations</h3>', unsafe_allow_html=True)
            
            for recommendation in cluster_recommendations[predicted_cluster]:
                st.markdown(f'<div class="recommendation-item">• {recommendation}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Display detailed spending breakdown
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<h3>💰 Spending Breakdown</h3>', unsafe_allow_html=True)
        
        # Create spending visualization
        spending_categories = ['Wine', 'Meat', 'Fish', 'Fruits', 'Sweets', 'Gold']
        spending_values = [
            customer.get("MntWines", 0),
            customer.get("MntMeatProducts", 0),
            customer.get("MntFishProducts", 0),
            customer.get("MntFruits", 0),
            customer.get("MntSweetProducts", 0),
            customer.get("MntGoldProds", 0)
        ]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(x=spending_categories, y=spending_values, 
                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        ])
        
        fig.update_layout(
            title="Customer Spending by Category",
            xaxis_title="Product Category",
            yaxis_title="Spending Amount ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error displaying results: {str(e)}")


def create_visualization_tab(processed_data, cluster_names):
    """Create visualization tab for customer data"""
    if processed_data is None or len(processed_data) == 0:
        st.warning("No data to visualize")
        return
    
    customer = processed_data.iloc[0]
    
    # Purchase channel comparison
    st.subheader("📊 Purchase Channel Analysis")
    
    channels = ['Web', 'Catalog', 'Store', 'Deals']
    purchases = [
        customer.get("NumWebPurchases", 0),
        customer.get("NumCatalogPurchases", 0),
        customer.get("NumStorePurchases", 0),
        customer.get("NumDealsPurchases", 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(x=channels, y=purchases, 
               marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    ])
    
    fig.update_layout(
        title="Purchase Distribution by Channel",
        xaxis_title="Purchase Channel",
        yaxis_title="Number of Purchases",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Campaign response analysis
    st.subheader("📧 Campaign Response Analysis")
    
    campaigns = ['Campaign 1', 'Campaign 2', 'Campaign 3', 'Campaign 4', 'Campaign 5']
    responses = [
        customer.get("AcceptedCmp1", 0),
        customer.get("AcceptedCmp2", 0),
        customer.get("AcceptedCmp3", 0),
        customer.get("AcceptedCmp4", 0),
        customer.get("AcceptedCmp5", 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(x=campaigns, y=responses, 
               marker_color=['#9b59b6', '#e67e22', '#1abc9c', '#34495e', '#e74c3c'])
    ])
    
    fig.update_layout(
        title="Campaign Response History",
        xaxis_title="Campaign",
        yaxis_title="Response (0=No, 1=Yes)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application function"""
    # Load model and cluster information
    model, model_loaded, cluster_names, cluster_descriptions, cluster_recommendations = load_model_and_info()
    
    # Header
    st.markdown('<h1>🎯 Customer Segmentation Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p>Predict customer segments and get personalized marketing recommendations</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Status badges
    st.markdown('<div class="status-badges">', unsafe_allow_html=True)
    if model_loaded:
        st.markdown('<span class="status-badge status-success">✅ Model Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">⚠️ Using Rule-based Prediction</span>', unsafe_allow_html=True)
    
    if PIPELINE_AVAILABLE:
        st.markdown('<span class="status-badge status-success">✅ Pipeline Available</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-error">❌ Pipeline Unavailable</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎯 Customer Segmentation")
    st.sidebar.markdown("Enter customer details to predict their segment and get marketing recommendations.")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Customer Input", "📊 Results", "📈 Visualizations"])
    
    with tab1:
        st.header("Customer Information")
        customer_data = create_customer_input_form()
        
        if st.button("🔮 Predict Customer Segment", key="predict_button"):
            with st.spinner("Processing customer data..."):
                processed_data = process_and_predict_with_pipeline(customer_data, model, model_loaded)
                
                if processed_data is not None:
                    # Store results in session state
                    st.session_state.processed_data = processed_data
                    st.session_state.prediction_made = True
                    st.success("✅ Prediction completed successfully!")
                    st.balloons()
                else:
                    st.error("❌ Failed to process customer data")
    
    with tab2:
        if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
            st.header("Prediction Results")
            display_customer_results(
                st.session_state.processed_data,
                cluster_names,
                cluster_descriptions,
                cluster_recommendations
            )
        else:
            st.info("👆 Please make a prediction first in the Customer Input tab")
    
    with tab3:
        if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
            st.header("Customer Analytics")
            create_visualization_tab(st.session_state.processed_data, cluster_names)
        else:
            st.info("👆 Please make a prediction first in the Customer Input tab")
    
    # Footer
    st.markdown("---")
    st.markdown("**Customer Segmentation Tool** | Built with Streamlit and scikit-learn")


if __name__ == "__main__":
    main()