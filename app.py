import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration Constants ---
# CRITICAL: SET THIS TO THE OPTIMAL THRESHOLD FOUND IN YOUR TUNING STEP (e.g., 0.525, 0.58, etc.)
OPTIMAL_THRESHOLD = 0.525 

# --- Load Model and Scaler ---
try:
    # Load the trained Logistic Regression model
    model = joblib.load('churn_model.pkl')
    # Load the StandardScaler (fitted on 1 feature at a time)
    scaler = joblib.load('feature_scaler.pkl') 
except FileNotFoundError:
    st.error("Model or Scaler files not found. Please ensure 'churn_model.pkl' and 'feature_scaler.pkl' are in the same directory.")
    st.stop()

# --- Feature Mappings and Definitions ---

# Mapped values for user input (must include all options presented to the user)
ALL_COUNTRIES = ['US', 'UK', 'Germany', 'Canada', 'Australia', 'France', 'India', 'Pakistan']
ALL_SUB_TYPES = ['Premium', 'Student', 'Family', 'Free']
ALL_DEVICES = ['Mobile', 'Desktop', 'Web']
ALL_GENDERS = ['Male', 'Female']

# List of columns that were Standard Scaled (must match the training order)
COLS_TO_STANDARD_SCALE = ['age', 'listening_time', 'songs_played_per_day', 'ads_listened_per_week']

# List of all expected features for the model input (X)
# ***FINAL ORDERING FIX***: Numerical/Binary first, followed by ALL OHE features sorted STRICTLY alphabetically 
# by the full feature name (e.g., country_AU, country_DE, ..., device_type_Desktop, ..., gender_Female, ...).
# FEATURE_COLUMNS = [
#    'age' 'listening_time' 'songs_played_per_day' 'ads_listened_per_week'
#  'offline_listening' 'gender_Female' 'gender_Male' 'country_AU'
#  'country_DE' 'country_FR' 'country_IN' 'country_PK' 'country_UK'
#  'country_US' 'subscription_type_Family' 'subscription_type_Premium'
#  'subscription_type_Student' 'device_type_Desktop' 'device_type_Mobile'
#  'skip_rate_scaled'
# ]
FEATURE_COLUMNS = list(model.feature_names_in_)
# --- Streamlit App Layout ---
st.set_page_config(page_title="Spotify Churn Predictor", layout="wide")
st.title("🎧 Spotify User Churn Prediction")
st.markdown("---")
st.markdown("Enter the user's current behavior metrics to assess their likelihood of churning.")

with st.form("churn_form"):
    st.header("User Data Input")

    col1, col2 = st.columns(2)

    with col1:
        # Numerical Inputs (Scaled)
        age = st.slider("Age", 18, 80, 30)
        listening_time = st.number_input("Avg. Weekly Listening Time (hours)", 1.0, 100.0, 15.0)
        songs_played_per_day = st.number_input("Songs Played Per Day", 1, 300, 50)
        
        # Binary Input
        offline_listening = st.selectbox("Offline Listening Enabled?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    with col2:
        # Categorical Inputs (Encoded)
        gender = st.selectbox("Gender", ALL_GENDERS)
        country = st.selectbox("Country", ALL_COUNTRIES)
        sub_type = st.selectbox("Subscription Type", ALL_SUB_TYPES)
        device = st.selectbox("Device Type", ALL_DEVICES)
        
        # Numerical Input (Scaled)
        skip_rate = st.slider("Skip Rate (Proportion of skipped songs, 0.0 to 0.6)", 0.0, 0.6, 0.2, step=0.01)
        ads_listened_per_week = st.number_input("Ads Listened Per Week", 0, 500, 10)

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    # --- 3. Preprocessing the User Input ---
    
    raw_features = {
        'age': age, 'listening_time': listening_time, 'songs_played_per_day': songs_played_per_day,
        'ads_listened_per_week': ads_listened_per_week, 'skip_rate': skip_rate,
        'offline_listening': offline_listening, 'gender': gender, 'country': country,
        'subscription_type': sub_type, 'device_type': device
    }
    
    input_df = pd.DataFrame([raw_features])
    
    # Scaling (Individual scaling for each column)
    for col in COLS_TO_STANDARD_SCALE:
        # Transform the single column as a 2D NumPy array
        input_df[col] = scaler.transform(input_df[[col]].values)[0, 0]
    
    # MinMax Scaling (skip_rate: max value assumed to be 0.6 from training)
    input_df['skip_rate_scaled'] = input_df['skip_rate'] / 0.6 

    # --- Create Final Feature Vector (Mandatory Order and Names) ---
    # Create a Series initialized to zeros with the correct feature names and order
    final_input = pd.Series(0, index=FEATURE_COLUMNS, dtype='float64')

    # Assign all scaled/binary values (6 features)
    final_input['age'] = input_df['age'][0]
    final_input['listening_time'] = input_df['listening_time'][0]
    final_input['songs_played_per_day'] = input_df['songs_played_per_day'][0]
    final_input['ads_listened_per_week'] = input_df['ads_listened_per_week'][0]
    final_input['skip_rate_scaled'] = input_df['skip_rate_scaled'][0]
    final_input['offline_listening'] = input_df['offline_listening'][0]
    
    # --- One-Hot Encoding Logic (Handling Dropped Baselines) ---
    
    # Gender (gender_Other is the dropped baseline)
    if gender == 'Female': final_input['gender_Female'] = 1
    elif gender == 'Male': final_input['gender_Male'] = 1
    
    # Country (country_CA is the dropped baseline)
    country_col = f'country_{country}'
    if country_col in final_input.index:
         final_input[country_col] = 1

    # Subscription Type (subscription_type_Free is the dropped baseline)
    if sub_type != 'Free':
        sub_col = f'subscription_type_{sub_type}'
        if sub_col in final_input.index:
            final_input[sub_col] = 1
    
    # Device Type (device_type_Web is the dropped baseline)
    if device != 'Web':
        device_col = f'device_type_{device}'
        if device_col in final_input.index:
             final_input[device_col] = 1
    
    # Convert the final Series into the 1xN DataFrame the model expects
    # The key is that final_input already has the correct names and order (from FEATURE_COLUMNS)
    model_input = final_input.to_frame().T
    
    # --- 4. Prediction and Output ---
    
    churn_prob = model.predict_proba(model_input)[0, 1]
    prediction = 1 if churn_prob >= OPTIMAL_THRESHOLD else 0
    
    st.subheader("Prediction Result")
    prob_percent = churn_prob * 100
    
    # Display results
    if prediction == 1:
        st.error(f"⚠️ HIGH CHURN RISK: The model predicts this user will churn.")
        st.metric("Churn Probability", f"{prob_percent:.1f}%", delta_color="off")
        st.write(f"This outcome is based on the optimal threshold of $\geq {OPTIMAL_THRESHOLD}$.")
    else:
        st.success(f"✅ LOW CHURN RISK: The model predicts this user will NOT churn.")
        st.metric("Churn Probability", f"{prob_percent:.1f}%", delta_color="off")
        st.write(f"This outcome is based on the optimal threshold of $< {OPTIMAL_THRESHOLD}$.")
        
    st.markdown("---")
    st.caption("Deployment model is Optimized Logistic Regression with Class Weights.")
