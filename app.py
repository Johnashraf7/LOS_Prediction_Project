import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Page config MUST be the first Streamlit command ────────────────────────────
st.set_page_config(
    page_title="LOS Predictor - Hospital Length of Stay",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for better styling ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .prediction-number {
        font-size: 5rem;
        font-weight: 800;
        margin: 1rem 0;
        line-height: 1;
    }
    
    .prediction-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
    }
    
    .stay-category {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
        padding: 0.5rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.2);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
        margin: 1rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Risk factor chips */
    .risk-chip {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Model loader with column extraction ────────────────────────────────────────
@st.cache_resource
def load_model_and_features():
    """Load the CatBoost model and extract expected features"""
    model_path = Path(__file__).resolve().parent / "Catboost_Model_LOS.pkl"

    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        return None, None

    load_errors = []

    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(model_path))

        expected_features = None
        try:
            if hasattr(model, 'feature_names_'):
                expected_features = model.feature_names_
            elif hasattr(model, 'get_feature_names'):
                expected_features = model.get_feature_names()
        except Exception:
            pass

        return model, expected_features
    except Exception as exc:
        load_errors.append(f"CatBoost load failed: {exc}")

    try:
        import joblib
        model = joblib.load(model_path)
        return model, None
    except Exception as exc:
        load_errors.append(f"joblib load failed: {exc}")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except Exception as exc:
        load_errors.append(f"pickle load failed: {exc}")

    st.error("Failed to load the model file.")
    st.code("`n".join(load_errors))
    return None, None


# ── Helper functions ───────────────────────────────────────────────────────────
def predict_los(model, input_data, expected_features=None):
    """Make prediction and return with interpretation"""
    try:
        # Ensure all expected columns are present
        if expected_features:
            # Check which expected features are missing
            missing_cols = set(expected_features) - set(input_data.columns)
            if missing_cols:
                st.warning(f"Adding missing columns with default values: {missing_cols}")
                for col in missing_cols:
                    input_data[col] = 0
            
            # Reorder columns to match model expectations
            input_data = input_data[expected_features]
        
        prediction = model.predict(input_data)[0]
        # Round to nearest whole number for days
        los = max(0, round(float(prediction)))  # Removed decimal, now whole number
        
        # Categorize LOS
        if los <= 3:
            category = "Short Stay"
            description = "Routine admission, expected quick recovery"
            icon = "🟢"
        elif los <= 7:
            category = "Moderate Stay"
            description = "Standard inpatient care duration"
            icon = "🟡"
        elif los <= 14:
            category = "Extended Stay"
            description = "Complex condition requiring longer monitoring"
            icon = "🟠"
        else:
            category = "Prolonged Stay"
            description = "Severe condition requiring intensive care"
            icon = "🔴"
        
        return los, category, description, icon
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("💡 Tip: Check if all required fields are filled correctly")
        return None, None, None, None

def get_risk_factors(data):
    """Extract and return active risk factors"""
    risk_mapping = {
        'dialysisrenalendstage': 'Dialysis/Renal Failure',
        'asthma': 'Asthma',
        'irondef': 'Iron Deficiency',
        'pneum': 'Pneumonia',
        'substancedependence': 'Substance Dependence',
        'psychologicaldisordermajor': 'Major Psychiatric Disorder',
        'depress': 'Depression',
        'psychother': 'Other Psychiatric Disorder',
        'fibrosisandother': 'Fibrosis/Pulmonary Disease',
        'malnutrition': 'Malnutrition'
    }
    
    active_risks = []
    for key, label in risk_mapping.items():
        if key in data and data[key] == 1:
            active_risks.append(label)
    
    return active_risks

# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Hospital Length of Stay Predictor</h1>
        <p>AI-powered prediction using CatBoost regression model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, expected_features = load_model_and_features()
    
    if model is None:
        st.stop()
    
    # Display expected features if available (for debugging)
    if expected_features and st.sidebar.checkbox("Show Model Info", value=False):
        with st.sidebar.expander("📊 Model Features"):
            st.write(f"Expected {len(expected_features)} features:")
            st.write(expected_features)
    
    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏥 Patient Information")
        st.markdown("---")
        
        # Demographics Section
        st.markdown("### 👤 Demographics")
        gender = st.selectbox("Gender", ["F", "M"], help="Patient's biological sex")
        
        rcount = st.selectbox(
            "Readmission Count (last 180 days)",
            ["0", "1", "2", "3", "4", "5+"],
            help="Number of prior hospital readmissions"
        )
        
        # BMI Input
        st.markdown("### 📏 Body Metrics")
        col1, col2 = st.columns(2)
        with col1:
            height_ft = st.number_input("Height (feet)", min_value=0.0, max_value=None, value=5.5, step=0.1, help="Height in feet")
        with col2:
            height_in = st.number_input("Height (inches)", min_value=0, max_value=None, value=0, step=1, help="Additional inches")
        
        weight_lbs = st.number_input("Weight (lbs)", min_value=0.0, max_value=None, value=150.0, step=1.0, help="Weight in pounds")
        
        # Calculate BMI
        height_total_inches = (height_ft * 12) + height_in
        if height_total_inches > 0:
            bmi = (weight_lbs / (height_total_inches ** 2)) * 703
            bmi = round(bmi, 1)
        else:
            bmi = 0.0
        
        # BMI Status (only show if valid)
        if bmi > 0:
            if bmi < 18.5:
                bmi_status = "⚠️ Underweight"
            elif bmi < 25:
                bmi_status = "✅ Normal"
            elif bmi < 30:
                bmi_status = "⚠️ Overweight"
            else:
                bmi_status = "⚠️ Obese"
            st.info(f"📊 BMI: **{bmi}** ({bmi_status})")
        else:
            st.info(f"📊 BMI: **{bmi}** (Invalid height/weight)")
        
        st.markdown("---")
        
        # Lab Results Section - NO RESTRICTIONS
        st.markdown("### 🧪 Lab Results")
        col1, col2 = st.columns(2)
        with col1:
            hematocrit = st.number_input("Hematocrit (%)", value=36.0, step=0.5, help="Red blood cell volume percentage")
            sodium = st.number_input("Sodium (mEq/L)", value=138.0, step=0.5, help="Serum sodium level")
            glucose = st.number_input("Glucose (mg/dL)", value=110.0, step=1.0, help="Blood glucose level")
            creatinine = st.number_input("Creatinine (mg/dL)", value=1.0, step=0.05, help="Kidney function marker")
        
        with col2:
            neutrophils = st.number_input("Neutrophils (×10³/µL)", value=6.0, step=0.1, help="White blood cell subtype")
            bloodureanitro = st.number_input("BUN (mg/dL)", value=18.0, step=0.5, help="Blood urea nitrogen level")
            hemo = st.number_input("Hemoglobin (g/dL)", value=12.5, step=0.1, help="Oxygen-carrying protein")
        
        st.markdown("---")
        
        # Vital Signs - NO RESTRICTIONS
        st.markdown("### 💓 Vital Signs")
        col1, col2 = st.columns(2)
        with col1:
            pulse = st.number_input("Heart Rate (bpm)", value=80, step=1, help="Beats per minute")
        with col2:
            respiration = st.number_input("Respiration Rate", value=16.0, step=0.5, help="Breaths per minute")
        
        st.markdown("---")
        
        # Comorbidities Section
        st.markdown("### 🩺 Comorbidities")
        st.caption("Select 1 if present, 0 if absent")
        
        col1, col2 = st.columns(2)
        with col1:
            dialysis = st.selectbox("Dialysis/Renal", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
            asthma = st.selectbox("Asthma", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
            irondef = st.selectbox("Iron Deficiency", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
            pneum = st.selectbox("Pneumonia", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
            substance = st.selectbox("Substance Dependence", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
        
        with col2:
            psychmajor = st.selectbox("Major Psychiatric", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
            depress = st.selectbox("Depression", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
            psychother = st.selectbox("Other Psychiatric", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
            fibrosis = st.selectbox("Fibrosis/Pulmonary", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
            malnutrition = st.selectbox("Malnutrition", [0, 1], format_func=lambda x: "✓ Present" if x else "✗ Absent")
        
        st.markdown("---")
        
        # Predict Button
        predict_button = st.button("🔮 PREDICT LENGTH OF STAY", type="primary", use_container_width=True)
    
    # ── MAIN CONTENT AREA ──────────────────────────────────────────────────────
    # Create two columns for main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">📋 Patient Summary</div>', unsafe_allow_html=True)
        
        # Create summary dataframe
        rcount_display = rcount
        
        # Format BMI display
        bmi_display = f"{bmi}" if bmi > 0 else "Invalid"
        if bmi > 0:
            bmi_display = f"{bmi} ({bmi_status})"
        
        summary_data = {
            "Category": ["Demographics", "Demographics", "Demographics",
                        "Labs", "Labs", "Labs", "Labs", "Labs", "Labs", "Labs",
                        "Vitals", "Vitals",
                        "Comorbidities", "Comorbidities", "Comorbidities", 
                        "Comorbidities", "Comorbidities", "Comorbidities", 
                        "Comorbidities", "Comorbidities", "Comorbidities", "Comorbidities"],
            "Parameter": ["Gender", "Readmission Count", "BMI",
                         "Hematocrit", "Neutrophils", "Sodium", "Glucose", 
                         "BUN", "Creatinine", "Hemoglobin",
                         "Heart Rate", "Respiration",
                         "Dialysis/Renal", "Asthma", "Iron Deficiency", "Pneumonia",
                         "Substance Dependence", "Major Psychiatric", "Depression",
                         "Other Psychiatric", "Fibrosis", "Malnutrition"],
            "Value": [gender, rcount_display, bmi_display,
                     f"{hematocrit:.1f} %", f"{neutrophils:.1f} ×10³/µL", 
                     f"{sodium:.1f} mEq/L", f"{glucose:.1f} mg/dL",
                     f"{bloodureanitro:.1f} mg/dL", f"{creatinine:.2f} mg/dL",
                     f"{hemo:.1f} g/dL",
                     f"{pulse} bpm", f"{respiration:.1f} /min",
                     "✓" if dialysis else "✗", "✓" if asthma else "✗",
                     "✓" if irondef else "✗", "✓" if pneum else "✗",
                     "✓" if substance else "✗", "✓" if psychmajor else "✗",
                     "✓" if depress else "✗", "✓" if psychother else "✗",
                     "✓" if fibrosis else "✗", "✓" if malnutrition else "✗"]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True, height=550)
    
    with col2:
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
        
        if predict_button:
            # Prepare input data for model
            rcount_num = int(rcount.replace('5+', '5'))
            
            input_dict = {
                "rcount": rcount_num,
                "gender": gender,
                "bmi": bmi if bmi > 0 else 25.0,  # Use default if BMI invalid
                "dialysisrenalendstage": dialysis,
                "asthma": asthma,
                "irondef": irondef,
                "pneum": pneum,
                "substancedependence": substance,
                "psychologicaldisordermajor": psychmajor,
                "depress": depress,
                "psychother": psychother,
                "fibrosisandother": fibrosis,
                "malnutrition": malnutrition,
                "hemo": hemo,
                "hematocrit": hematocrit,
                "neutrophils": neutrophils,
                "sodium": sodium,
                "glucose": glucose,
                "bloodureanitro": bloodureanitro,
                "creatinine": creatinine,
                "pulse": pulse,
                "respiration": respiration
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Make prediction
            los, category, description, icon = predict_los(model, input_df, expected_features)
            
            if los is not None:
                # Display prediction card with whole number
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">PREDICTED LENGTH OF STAY</div>
                    <div class="prediction-number">{los}</div>
                    <div style="font-size: 1.2rem; margin-top: -0.5rem;">days</div>
                    <div class="stay-category">
                        {icon} {category} {icon}
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.9rem;">
                        {description}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors
                active_risks = get_risk_factors(input_dict)
                if active_risks:
                    st.markdown("#### ⚠️ Active Risk Factors")
                    risk_chips = "".join(f'<span class="risk-chip">{risk}</span>' for risk in active_risks)
                    st.markdown(risk_chips, unsafe_allow_html=True)
                    
                    risk_count = len(active_risks)
                    if risk_count >= 3:
                        st.warning(f"⚠️ {risk_count} comorbidities detected. Consider enhanced care coordination.")
                
                # BMI insights (only if valid)
                if bmi > 0:
                    if bmi < 18.5:
                        st.warning("📉 Underweight BMI - Monitor nutritional status")
                    elif bmi > 30:
                        st.warning("📈 Obese BMI - Consider weight management and associated risks")
                    elif bmi > 25:
                        st.info("📊 Overweight BMI - Monitor for metabolic complications")
                
                # Clinical recommendations based on LOS
                st.markdown("#### 💡 Clinical Insights")
                if los <= 3:
                    st.info("""
                    **Recommendations:**
                    - Routine discharge planning
                    - Standard post-discharge follow-up
                    - Monitor for early readmission risk factors
                    """)
                elif los <= 7:
                    st.info("""
                    **Recommendations:**
                    - Coordinate with case management
                    - Ensure discharge criteria are met
                    - Schedule follow-up within 7 days
                    """)
                elif los <= 14:
                    st.warning("""
                    **Recommendations:**
                    - Involve multidisciplinary team
                    - Consider rehabilitation services
                    - Detailed discharge planning required
                    """)
                else:
                    st.error("""
                    **Recommendations:**
                    - Intensive case management needed
                    - Consider skilled nursing facility
                    - Complex care coordination required
                    - Regular team meetings recommended
                    """)
                
                # Visual indicator
                st.markdown("#### 📊 Stay Duration Indicator")
                progress_value = min(los / 21, 1.0)
                st.progress(progress_value, text=f"Stay duration: {los} days")
                
        else:
            st.info("👈 **Ready for prediction**\n\nFill in all patient information in the sidebar and click 'Predict Length of Stay' to see results.")
            
            # Show example or demo message
            st.markdown("""
            <div class="info-box">
                <strong>📌 How to use:</strong><br>
                1. Enter patient demographics including height/weight for BMI<br>
                2. Input lab results and vital signs (any values allowed)<br>
                3. Select comorbidities (1 = present, 0 = absent)<br>
                4. Click the predict button to see results<br>
                5. Review risk factors and clinical insights
            </div>
            """, unsafe_allow_html=True)
            
            # Display model information
            with st.expander("ℹ️ About the Model"):
                st.markdown("""
                **Model Details:**
                - **Algorithm:** CatBoost Regressor
                - **Features:** 22 clinical parameters
                - **Target:** Length of Stay (days)
                - **Training Data:** Historical hospital admissions
                
                **Key Features:**
                - Demographics (gender, BMI, readmission count)
                - Lab values (CBC, metabolic panel)
                - Vital signs (heart rate, respiration)
                - Comorbidities (10 conditions)
                
                **BMI Categories:**
                - Underweight: < 18.5
                - Normal: 18.5 - 24.9
                - Overweight: 25 - 29.9
                - Obese: ≥ 30
                
                **Output Format:**
                - Predicted length of stay is rounded to the nearest whole day
                - Categories help interpret the expected care intensity
                
                **Note:** Lab values and vital signs can accept any numeric value with no restrictions.
                """)
    
    # Footer disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <small>
        ⚠️ <strong>Clinical Disclaimer:</strong> This tool is for informational and research purposes only. 
        Predictions are based on statistical models and should not replace clinical judgment. 
        Always consult with qualified healthcare professionals for medical decisions.
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

