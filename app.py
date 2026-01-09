import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘")
st.title("🫘 Dry Bean Variety Classifier")

# Load Assets
if os.path.exists('bean_model.pkl'):
    model = joblib.load('bean_model.pkl')
    scaler = joblib.load('bean_scaler.pkl')
    le = joblib.load('bean_encoder.pkl')

    st.write("Enter bean morphology features to identify the variety.")

    with st.form("bean_form"):
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Area", value=40000.0)
            perim = st.number_input("Perimeter", value=800.0)
            major = st.number_input("Major Axis Length", value=300.0)
            minor = st.number_input("Minor Axis Length", value=200.0)
            aspect = st.number_input("Aspect Ratio", value=1.5)
            eccen = st.number_input("Eccentricity", value=0.7)
            conv_area = st.number_input("Convex Area", value=41000.0)
            equiv_dia = st.number_input("Equiv Diameter", value=230.0)
        
        with col2:
            extent = st.number_input("Extent", value=0.7)
            solidity = st.number_input("Solidity", value=0.9)
            roundness = st.number_input("Roundness", value=0.8)
            compact = st.number_input("Compactness", value=0.8)
            shape1 = st.number_input("Shape Factor 1", format="%.5f", value=0.006)
            shape2 = st.number_input("Shape Factor 2", format="%.5f", value=0.002)
            shape3 = st.number_input("Shape Factor 3", format="%.5f", value=0.6)
            shape4 = st.number_input("Shape Factor 4", format="%.5f", value=0.9)

        if st.form_submit_button("Classify Bean"):
            # Prepare all 16 features in order
            raw_features = np.array([[area, perim, major, minor, aspect, eccen, conv_area, 
                                     equiv_dia, extent, solidity, roundness, compact, 
                                     shape1, shape2, shape3, shape4]])
            
            scaled_features = scaler.transform(raw_features)
            prediction_id = model.predict(scaled_features)
            bean_type = le.inverse_transform(prediction_id)[0]
            
            st.success(f"### Predicted Variety: **{bean_type}**")
else:
    st.error("Missing model files! Please upload 'bean_model.pkl', 'bean_scaler.pkl', and 'bean_encoder.pkl' to GitHub.")