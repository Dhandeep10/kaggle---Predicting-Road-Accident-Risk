import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="🚗 Road Accident Risk Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load('model_xgb.pkl')
        lgb_model = joblib.load('model_lgb.pkl')
        cat_model = joblib.load('model_cat.pkl')
        et_model = joblib.load('model_et.pkl')
        rf_model = joblib.load('model_rf.pkl')
        meta_model = joblib.load('meta.pkl')
        st.sidebar.success("✅ All models loaded!")
        return xgb_model, lgb_model, cat_model, et_model, rf_model, meta_model
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        st.stop()

xgb_model, lgb_model, cat_model, et_model, rf_model, meta_model = load_models()

# Title
st.title("🚗 Road Accident Risk Predictor")
st.markdown("### Kaggle Competition Winner - Full Ensemble System")

# Sidebar inputs
st.sidebar.title("📋 Road Parameters")
st.sidebar.markdown("---")

# Basic road features
st.sidebar.markdown("### 🛣️ Road Characteristics")
road_type = st.sidebar.selectbox("Road Type", [0, 1, 2, 3], format_func=lambda x: ["Highway", "Urban", "Rural", "Residential"][x])
num_lanes = st.sidebar.slider("Number of Lanes", 1, 6, 2)
curvature = st.sidebar.slider("Road Curvature", 0.0, 2.0, 0.5, 0.1)
speed_limit = st.sidebar.slider("Speed Limit (km/h)", 20, 120, 60, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌤️ Environmental Conditions")
lighting = st.sidebar.selectbox("Lighting", [0, 1, 2], format_func=lambda x: ["Daylight", "Twilight", "Dark"][x])
weather = st.sidebar.selectbox("Weather", [0, 1, 2], format_func=lambda x: ["Clear", "Rain", "Fog/Snow"][x])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚦 Safety Features")
road_signs_present = st.sidebar.checkbox("Road Signs Present", value=True)
public_road = st.sidebar.checkbox("Public Road", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏰ Time Factors")
time_of_day = st.sidebar.selectbox("Time of Day", [0, 1, 2, 3], format_func=lambda x: ["Morning", "Afternoon", "Evening", "Night"][x])
holiday = st.sidebar.checkbox("Holiday")
school_season = st.sidebar.checkbox("School Season", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Historical Data")
num_reported_accidents = st.sidebar.number_input("Historical Accidents", 0, 100, 5)

predict_button = st.sidebar.button("🎯 PREDICT RISK", type="primary", use_container_width=True)

# Feature engineering (matching your Kaggle notebook)
def create_features(road_type, num_lanes, curvature, speed_limit, lighting, weather,
                    road_signs_present, public_road, time_of_day, holiday, 
                    school_season, num_reported_accidents):
    
    features = {
        'road_type': road_type,
        'num_lanes': num_lanes,
        'curvature': curvature,
        'speed_limit': speed_limit,
        'lighting': lighting,
        'weather': weather,
        'road_signs_present': int(road_signs_present),
        'public_road': int(public_road),
        'time_of_day': time_of_day,
        'holiday': int(holiday),
        'school_season': int(school_season),
        'num_reported_accidents': num_reported_accidents,
        
        # Engineered features (matching your training)
        'speed_curvature': speed_limit * curvature,
        'speed_curvature_sq': speed_limit * curvature ** 2,
        'lanes_speed': num_lanes * speed_limit,
        'accidents_speed': num_reported_accidents * speed_limit,
        'curvature_lanes': curvature * num_lanes,
        'accidents_curvature': num_reported_accidents * curvature,
        'curvature_sq': curvature ** 2,
        'curvature_cb': curvature ** 3,
        'curvature_sqrt': np.sqrt(curvature),
        'speed_sq': speed_limit ** 2,
        'speed_sqrt': np.sqrt(speed_limit),
        'accidents_sq': num_reported_accidents ** 2,
        'accidents_per_lane': num_reported_accidents / (num_lanes + 1),
        'speed_per_lane': speed_limit / (num_lanes + 1),
        'curvature_per_lane': curvature / (num_lanes + 1),
        'speed_curv_lanes': speed_limit * curvature * num_lanes,
        'speed_curv_acc': speed_limit * curvature * num_reported_accidents,
        'danger_score': (speed_limit / 50) * (curvature * 2) * (num_reported_accidents / 10),
        'risky_weather': int(weather >= 1),
        'poor_lighting': int(lighting >= 1),
        'high_speed': int(speed_limit > 80),
        'sharp_curve': int(curvature > 1.0),
        'many_accidents': int(num_reported_accidents > 10),
        'extreme_risk': int((weather >= 1) and (curvature > 1.0) and (speed_limit > 80)),
        'safe_combo': int((weather == 0) and (lighting == 0) and (road_signs_present == 1)),
        'rush_hour_danger': int((time_of_day in [1, 2]) and (num_lanes >= 3)),
        'school_risk': int(school_season and (time_of_day in [0, 2])),
        'speed_bin': int(speed_limit // 20),
        'curve_bin': int(curvature // 0.5),
        'acc_bin': int(num_reported_accidents // 5),
        'road_weather': road_type * 10 + weather,
        'road_lighting': road_type * 10 + lighting,
        'weather_lighting': weather * 10 + lighting,
        'road_time': road_type * 10 + time_of_day,
        'weather_time': weather * 10 + time_of_day,
        'lighting_time': lighting * 10 + time_of_day
    }
    
    return pd.DataFrame([features])

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Input Summary")
    summary = pd.DataFrame({
        'Parameter': ['Road Type', 'Lanes', 'Curvature', 'Speed Limit', 'Lighting', 'Weather', 'Signs', 'Accidents'],
        'Value': [
            ["Highway", "Urban", "Rural", "Residential"][road_type],
            str(num_lanes),
            f"{curvature:.1f}",
            f"{speed_limit} km/h",
            ["Daylight", "Twilight", "Dark"][lighting],
            ["Clear", "Rain", "Fog/Snow"][weather],
            "Yes" if road_signs_present else "No",
            str(num_reported_accidents)
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

with col2:
    if predict_button:
        # Create features
        features = create_features(road_type, num_lanes, curvature, speed_limit, lighting, weather,
                                   road_signs_present, public_road, time_of_day, holiday,
                                   school_season, num_reported_accidents)
        
        # Make predictions
        with st.spinner("🔄 Running 5-model ensemble..."):
            pred_xgb = xgb_model.predict(features)[0]
            pred_lgb = lgb_model.predict(features)[0]
            pred_cat = cat_model.predict(features)[0]
            pred_et = et_model.predict(features)[0]
            pred_rf = rf_model.predict(features)[0]
            
            # Stack and meta-predict
            stacked = np.array([[pred_xgb, pred_lgb, pred_cat, pred_et, pred_rf]])
            final_pred = meta_model.predict(stacked)[0]
            risk_score = np.clip(final_pred, 0, 1)
        
        # Display result
        if risk_score < 0.3:
            st.success(f"🟢 **LOW RISK** - Score: {risk_score:.4f}")
        elif risk_score < 0.6:
            st.warning(f"🟡 **MEDIUM RISK** - Score: {risk_score:.4f}")
        else:
            st.error(f"🔴 **HIGH RISK** - Score: {risk_score:.4f}")
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            title={'text': "Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual models
        with st.expander("🔬 Individual Model Predictions"):
            model_df = pd.DataFrame({
                'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'RandomForest', 'Meta'],
                'Prediction': [pred_xgb, pred_lgb, pred_cat, pred_et, pred_rf, risk_score]
            })
            st.dataframe(model_df, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center;'>🏆 Kaggle Score: 0.05556 | Rank: ~500-700</p>", unsafe_allow_html=True)
