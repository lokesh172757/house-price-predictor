import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Setup
st.set_page_config(page_title="AI House Appraiser", page_icon="🏠", layout="centered")

st.title("🏠 AI Real Estate Appraiser")

# --- Context Info for Users ---
st.info("""
**ℹ️ Data Source:** This model is trained on real estate data from **Ames, Iowa (USA)**. 
Predictions are in **USD ($)** based on 2010 market conditions.
*(INR conversion provided for reference)*
""")

st.write("Adjust the property details below to estimate market value.")

# 2. Load Assets (Model & Defaults)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('house_price_model.pkl')
        defaults = joblib.load('house_price_defaults.pkl')
        return model, defaults
    except FileNotFoundError:
        return None, None

model, defaults = load_assets()

if model is None:
    st.error("Error: Model files not found. Please run the training notebook first to generate the .pkl files!")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("Property Specs")

# Using defaults to set the starting points
gr_liv_area = st.sidebar.slider("Living Area (sq ft)", 500, 5000, int(defaults['GrLivArea']))
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, int(defaults['OverallQual']))
year_built = st.sidebar.slider("Year Built", 1900, 2024, int(defaults['YearBuilt']))
total_bsmt = st.sidebar.slider("Basement Size (sq ft)", 0, 3000, int(defaults['TotalBsmtSF']))
garage_cars = st.sidebar.slider("Garage Capacity", 0, 4, int(defaults['GarageCars']))

# Dropdown for Neighborhood (Actual values from the dataset)
neighborhoods = ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", 
                 "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", 
                 "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert", 
                 "StoneBr", "ClearCr", "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"]

neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods, index=0)

# 4. Logic: Combine Inputs with Defaults
# Start with the average house (fills all 80 columns)
input_data = defaults.copy()

# Update with user specific choices
input_data['GrLivArea'] = gr_liv_area
input_data['OverallQual'] = overall_qual
input_data['YearBuilt'] = year_built
input_data['TotalBsmtSF'] = total_bsmt
input_data['GarageCars'] = garage_cars
input_data['Neighborhood'] = neighborhood

# Create DataFrame (1 row, 80 columns)
df = pd.DataFrame([input_data])

# 5. Display Summary
st.subheader("Property Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Size", f"{gr_liv_area} sqft")
col2.metric("Quality", f"{overall_qual}/10")
col3.metric("Location", neighborhood)

# 6. Predict
if st.button("Estimate Price", type="primary"):
    with st.spinner("Analyzing market trends..."):
        # Predict Log Price (Pipeline handles scaling/encoding)
        log_prediction = model.predict(df)
        
        # Invert Log -> Real Dollars
        real_prediction = np.expm1(log_prediction)[0]
        
        # Convert to INR (Approx 1 USD = 83 INR)
        inr_price = real_prediction * 83
        
    st.markdown("---")
    st.markdown(f"### 💰 Estimated Value: **${real_prediction:,.2f}**")
    st.caption(f"*(Approx. ₹{inr_price/100000:,.2f} Lakhs INR)*")
    
    # Contextual Advice
    if overall_qual >= 8:
        st.success("💎 Premium Property: High quality construction significantly boosts valuation.")
    elif overall_qual <= 4:
        st.warning("🛠️ Fixer-Upper: Valuation assumes standard condition; renovation needs may lower actual price.")