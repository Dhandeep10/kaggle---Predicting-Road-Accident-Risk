import streamlit as st
import numpy as np
import pandas as pd
import joblib
from functools import lru_cache
from typing import Dict, Tuple

# Page config - FIRST
st.set_page_config(page_title="Road Safety App", page_icon="🛣️", layout="wide")

# Simple styling
st.markdown("""
<style>
.badge { padding: 6px 12px; border-radius: 20px; font-weight: bold; margin: 2px; display: inline-block; }
.stat { text-align: center; padding: 12px; background: #1a1a2e; border-radius: 8px; margin: 10px 0; color: white; }
.good { background: #2d5a27; color: #2ecc71; }
.warn { background: #4a4a1a; color: #f1c40f; }
.bad { background: #4a1a1a; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

st.title("🛣️ Road Safety – Game | Explore | Journey")

# Model loading
@st.cache_resource
def load_models():
    xgb = joblib.load("model_xgb.pkl")
    lgb = joblib.load("model_lgb.pkl")
    cat = joblib.load("model_cat.pkl")
    et = joblib.load("model_et.pkl")
    rf = joblib.load("model_rf.pkl")
    meta = joblib.load("meta.pkl")
    return xgb, lgb, cat, et, rf, meta

try:
    xgb_model, lgb_model, cat_model, et_model, rf_model, meta_model = load_models()
    st.success("✅ Ensemble models loaded!")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

BASE_COLS = [
    "road_type","num_lanes","curvature","speed_limit","lighting","weather",
    "road_signs_present","public_road","time_of_day","holiday","school_season",
    "num_reported_accidents"
]

def create_features(base):
    rt,nl,cv,sl,lt,wt = base["road_type"], base["num_lanes"], base["curvature"], base["speed_limit"], base["lighting"], base["weather"]
    rs,pb,tod,hol,sch,acc = int(base["road_signs_present"]), int(base["public_road"]), base["time_of_day"], int(base["holiday"]), int(base["school_season"]), base["num_reported_accidents"]
    feats = {
        "road_type": rt, "num_lanes": nl, "curvature": cv, "speed_limit": sl,
        "lighting": lt, "weather": wt, "road_signs_present": rs, "public_road": pb,
        "time_of_day": tod, "holiday": hol, "school_season": sch, "num_reported_accidents": acc,
        "speed_curvature": sl*cv, "speed_curvature_sq": sl*(cv**2), "lanes_speed": nl*sl,
        "accidents_speed": acc*sl, "curvature_lanes": cv*nl, "accidents_curvature": acc*cv,
        "curvature_sq": cv**2, "curvature_cb": cv**3, "curvature_sqrt": np.sqrt(cv+1e-9),
        "speed_sq": sl**2, "speed_sqrt": np.sqrt(sl), "accidents_sq": acc**2,
        "accidents_per_lane": acc/(nl+1), "speed_per_lane": sl/(nl+1), "curvature_per_lane": cv/(nl+1),
        "speed_curv_lanes": sl*cv*nl, "speed_curv_acc": sl*cv*acc,
        "danger_score": (sl/50)*(2*cv)*(acc/10 if acc else 0),
        "risky_weather": int(wt>=1), "poor_lighting": int(lt>=1),
        "high_speed": int(sl>80), "sharp_curve": int(cv>1.0), "many_accidents": int(acc>10),
        "extreme_risk": int((wt>=1) and (cv>1.0) and (sl>80)),
        "safe_combo": int((wt==0) and (lt==0) and (rs==1)),
        "rush_hour_danger": int((tod in [1,2]) and (nl>=3)),
        "school_risk": int(sch and (tod in [0,2])),
        "speed_bin": int(sl//20), "curve_bin": int(cv//0.5), "acc_bin": int(acc//5),
        "road_weather": rt*10+wt, "road_lighting": rt*10+lt, "weather_lighting": wt*10+lt,
        "road_time": rt*10+tod, "weather_time": wt*10+tod, "lighting_time": lt*10+tod
    }
    return pd.DataFrame([feats])

@lru_cache(maxsize=2048)
def risk_and_parts_cached(road_tuple):
    base = {k: v for k, v in zip(BASE_COLS, road_tuple)}
    feats = create_features(base)
    p1 = float(xgb_model.predict(feats)[0])
    p2 = float(lgb_model.predict(feats)[0])
    p3 = float(cat_model.predict(feats)[0])
    p4 = float(et_model.predict(feats)[0])
    p5 = float(rf_model.predict(feats)[0])
    final = float(np.clip(meta_model.predict(np.array([[p1,p2,p3,p4,p5]]))[0], 0, 1))
    return final, (p1,p2,p3,p4,p5)

def risk_only(base):
    t = tuple(base[k] for k in BASE_COLS)
    y, _ = risk_and_parts_cached(t)
    return y

def risk_and_parts(base):
    t = tuple(base[k] for k in BASE_COLS)
    y, parts = risk_and_parts_cached(t)
    return y, {"XGBoost":parts[0],"LightGBM":parts[1],"CatBoost":parts[2],"ExtraTrees":parts[3],"RandomForest":parts[4]}

def random_road(rng):
    return {
        "road_type": int(rng.integers(0, 4)),
        "num_lanes": int(rng.integers(1, 7)),
        "curvature": float(np.round(rng.uniform(0.0, 2.0), 2)),
        "speed_limit": int(rng.choice([40, 60, 80, 100, 120])),
        "lighting": int(rng.integers(0, 3)),
        "weather": int(rng.integers(0, 3)),
        "road_signs_present": bool(rng.integers(0, 2)),
        "public_road": True,
        "time_of_day": int(rng.integers(0, 4)),
        "holiday": bool(rng.integers(0, 2)),
        "school_season": bool(rng.integers(0, 2)),
        "num_reported_accidents": int(rng.integers(0, 101)),
    }

def pretty_table(base):
    road_names = ["Highway","Urban","Rural","Residential"]
    light_names = ["Daylight","Twilight","Dark"]
    weather_names = ["Clear","Rain","Fog/Snow"]
    return pd.DataFrame({
        "Parameter": ["🛣️ Road Type","🛤️ Lanes","🌀 Curvature","🚗 Speed Limit",
                      "💡 Lighting","🌦️ Weather","🚸 Signs","📊 Past Accidents"],
        "Value": [
            road_names[base["road_type"]], f'{base["num_lanes"]} lanes', f'{base["curvature"]:.1f}/2.0',
            f'{base["speed_limit"]} km/h', light_names[base["lighting"]], weather_names[base["weather"]],
            "Present ✅" if base["road_signs_present"] else "Absent ❌", f'{base["num_reported_accidents"]} accidents'
        ]
    })

# Tabs
tab_game, tab_explore, tab_journey = st.tabs(["🎮 Game", "🔍 Explore", "🚦 Journey"])

# ======================
# GAME TAB
# ======================
with tab_game:
    # Game state
    if "game_active" not in st.session_state:
        st.session_state.game_active = True
        st.session_state.game_current_round = 1
        st.session_state.game_max_rounds = 10
        st.session_state.game_score = 0
        st.session_state.game_history = []
        st.session_state.game_session_id = str(np.random.randint(1000000, 9999999))

    def generate_game_roads(round_num):
        rng_a = np.random.default_rng(st.session_state.game_session_id + round_num)
        rng_b = np.random.default_rng(st.session_state.game_session_id + round_num + 100)
        return random_road(rng_a), random_road(rng_b)

    def reset_game():
        st.session_state.game_active = True
        st.session_state.game_current_round = 1
        st.session_state.game_score = 0
        st.session_state.game_history = []
        st.session_state.game_session_id = str(np.random.randint(1000000, 9999999))
        st.rerun()

    # Game summary (only show when game ends)
    if not st.session_state.game_active:
        st.header(f"🎯 Game Complete! Final Score: {st.session_state.game_score}/10")
        
        if st.session_state.game_history:
            # Simple summary without st.bullet()
            accuracy = len([h for h in st.session_state.game_history if h["correct"]]) / len(st.session_state.game_history) * 100
            st.markdown(f"**Overall Accuracy**: {accuracy:.1f}%")
            
            correct_count = sum(1 for h in st.session_state.game_history if h["correct"])
            if accuracy > 70:
                st.markdown("**Rating**: Expert Driver 🚀")
            elif accuracy > 50:
                st.markdown("**Rating**: Good Driver 👍")
            else:
                st.markdown("**Rating**: Need Practice 📚")

            # History table
            history_df = pd.DataFrame(st.session_state.game_history)
            st.subheader("Round Details")
            for idx, row in history_df.iterrows():
                status = "✅" if row["correct"] else "❌"
                st.write(f"Round {row['round']}: {status} (A: {row['risk_a']:.3f} vs B: {row['risk_b']:.3f})")

        if st.button("🎮 New Game", type="primary"):
            reset_game()
        st.stop()

    # Current round
    current_round = st.session_state.game_current_round
    st.header(f"Round {current_round}/10")

    # Generate roads for this specific round
    road_a, road_b = generate_game_roads(current_round)

    col1, col2 = st.columns(2)

    # Road A
    with col1:
        st.subheader("🛣️ Road A")
        st.dataframe(pretty_table(road_a), use_container_width=True, height=280)
        if st.button("🚗 Choose A", key=f"game_a_button_{st.session_state.game_session_id}_{current_round}"):
            risk_a = risk_only(road_a)
            risk_b = risk_only(road_b)
            correct = risk_a < risk_b
            st.session_state.game_history.append({
                "round": current_round,
                "choice": "A",
                "risk_a": risk_a,
                "risk_b": risk_b,
                "correct": correct
            })
            
            if correct:
                st.session_state.game_score += 1
                st.success(f"✅ Correct! Score: {st.session_state.game_score}")
            else:
                st.error(f"❌ Riskier choice. Score: {st.session_state.game_score}")
                st.write(f"Safer was Road B: A={risk_a:.3f}, B={risk_b:.3f}")

            st.session_state.game_current_round += 1
            if st.session_state.game_current_round > st.session_state.game_max_rounds:
                st.session_state.game_active = False
            st.rerun()

    # Road B
    with col2:
        st.subheader("🛣️ Road B")
        st.dataframe(pretty_table(road_b), use_container_width=True, height=280)
        if st.button("🚗 Choose B", key=f"game_b_button_{st.session_state.game_session_id}_{current_round}"):
            risk_a = risk_only(road_a)
            risk_b = risk_only(road_b)
            correct = risk_b < risk_a
            st.session_state.game_history.append({
                "round": current_round,
                "choice": "B",
                "risk_a": risk_a,
                "risk_b": risk_b,
                "correct": correct
            })
            
            if correct:
                st.session_state.game_score += 1
                st.success(f"✅ Correct! Score: {st.session_state.game_score}")
            else:
                st.error(f"❌ Riskier choice. Score: {st.session_state.game_score}")
                st.write(f"Safer was Road A: A={risk_a:.3f}, B={risk_b:.3f}")

            st.session_state.game_current_round += 1
            if st.session_state.game_current_round > st.session_state.game_max_rounds:
                st.session_state.game_active = False
            st.rerun()

    # Progress bar
    rounds_completed = len(st.session_state.game_history)
    progress = min(rounds_completed, st.session_state.game_max_rounds) / st.session_state.game_max_rounds
    st.progress(progress)

    # Score display
    col1, col2 = st.columns(2)
    col1.metric("Current Score", st.session_state.game_score)
    col2.metric("Round", f"{rounds_completed}/10")

# ======================
# EXPLORE TAB
# ======================
with tab_explore:
    st.subheader("🔍 Explore Road Risk")
    st.write("Configure parameters to see real-time risk prediction from ensemble models")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        road_type = st.selectbox("Road Type", range(4), 
                                format_func=lambda x: ["Highway","Urban","Rural","Residential"][x],
                                key="explore_road_type")
        num_lanes = st.slider("Number of Lanes", 1, 6, 2, key="explore_num_lanes")
        curvature = st.slider("Curvature (0-2)", 0.0, 2.0, 0.5, 0.1, key="explore_curvature")
        speed_limit = st.select_slider("Speed Limit", [40,60,80,100,120], value=80, key="explore_speed_limit")
    
    with col2:
        lighting = st.selectbox("Lighting", range(3), 
                               format_func=lambda x: ["Daylight","Twilight","Dark"][x],
                               key="explore_lighting")
        weather = st.selectbox("Weather", range(3), 
                              format_func=lambda x: ["Clear","Rain","Fog/Snow"][x],
                              key="explore_weather")
        time_of_day = st.selectbox("Time of Day", range(4),
                                  format_func=lambda x: ["Morning","Afternoon","Evening","Night"][x],
                                  key="explore_time_of_day")
        num_accidents = st.slider("Past Accidents (1 year)", 0, 100, 5, key="explore_num_accidents")

    # Additional controls
    col3, col4 = st.columns(2)
    with col3:
        road_signs = st.checkbox("Road Signs Present", True, key="explore_road_signs")
        public_road = st.checkbox("Public Road", True, key="explore_public_road")
    
    with col4:
        holiday = st.checkbox("Holiday Period", False, key="explore_holiday")
        school_season = st.checkbox("School Season", True, key="explore_school_season")

    # Build road configuration
    road_config = {
        "road_type": road_type,
        "num_lanes": num_lanes,
        "curvature": curvature,
        "speed_limit": speed_limit,
        "lighting": lighting,
        "weather": weather,
        "road_signs_present": road_signs,
        "public_road": public_road,
        "time_of_day": time_of_day,
        "holiday": holiday,
        "school_season": school_season,
        "num_reported_accidents": num_accidents
    }

    # Predict button
    col_pred, col_result = st.columns([1, 2])
    with col_pred:
        if st.button("🎯 Calculate Risk", type="primary", use_container_width=True):
            risk_score, model_parts = risk_and_parts(road_config)
            st.session_state.explore_results = {
                "risk_score": risk_score,
                "model_parts": model_parts,
                "config": road_config
            }

    # Display results
    if "explore_results" in st.session_state:
        results = st.session_state.explore_results
        risk_score = results["risk_score"]
        
        # Risk assessment
        if risk_score < 0.3:
            st.success(f"**Risk Level: LOW** ({risk_score:.3f})")
        elif risk_score < 0.6:
            st.warning(f"**Risk Level: MEDIUM** ({risk_score:.3f})")
        else:
            st.error(f"**Risk Level: HIGH** ({risk_score:.3f})")

        # Model contributions table
        models_df = pd.DataFrame({
            "Model": ["XGBoost", "LightGBM", "CatBoost", "ExtraTrees", "RandomForest"],
            "Risk Contribution": [
                results["model_parts"]["XGBoost"],
                results["model_parts"]["LightGBM"],
                results["model_parts"]["CatBoost"],
                results["model_parts"]["ExtraTrees"],
                results["model_parts"]["RandomForest"]
            ]
        })
        st.table(models_df)

    # Current configuration
    st.subheader("Road Configuration")
    config_df = pd.DataFrame({
        "Parameter": ["Road Type", "Lanes", "Curvature", "Speed Limit", "Lighting", "Weather", "Time of Day", "Road Signs", "Public Road", "Holiday", "School Season", "Past Accidents"],
        "Value": [
            ["Highway","Urban","Rural","Residential"][road_type],
            num_lanes,
            f"{curvature:.1f}/2.0",
            f"{speed_limit} km/h",
            ["Daylight","Twilight","Dark"][lighting],
            ["Clear","Rain","Fog/Snow"][weather],
            ["Morning","Afternoon","Evening","Night"][time_of_day],
            "Present" if road_signs else "Absent",
            "Yes" if public_road else "No",
            "Yes" if holiday else "No",
            "Yes" if school_season else "No",
            num_accidents
        ]
    })
    st.table(config_df)

# ======================
# JOURNEY TAB
# ======================
with tab_journey:
    # Journey state
    if "journey_step" not in st.session_state:
        st.session_state.journey_step = 1
        st.session_state.journey_total = 5
        st.session_state.journey_risk_threshold = 0.60
        st.session_state.journey_choices = []
        st.session_state.journey_active = True
        st.session_state.journey_id = np.random.randint(4000, 9999)

    def generate_journey_choices(step):
        rng = np.random.default_rng(st.session_state.journey_id + step * 100)
        choices = []
        for i in range(3):
            road = random_road(rng)
            risk = risk_only(road)
            choices.append({"road": road, "risk": risk})
        return choices

    def reset_journey():
        st.session_state.journey_step = 1
        st.session_state.journey_choices = []
        st.session_state.journey_active = True
        st.session_state.journey_id = np.random.randint(4000, 9999)
        st.rerun()

    # Journey controls
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.journey_total = st.slider("Journey Length", 3, 8, 5, key="journey_length")
        st.session_state.journey_risk_threshold = st.slider("Risk Threshold", 0.4, 0.9, 0.6, 0.05, key="journey_threshold")
    with col2:
        if st.button("🚀 Start New Journey", type="primary", use_container_width=True):
            reset_journey()

    # Journey complete
    if not st.session_state.journey_active:
        st.header("🏁 Journey Complete!")
        if len(st.session_state.journey_choices) == st.session_state.journey_total:
            st.success("Successfully navigated all steps!")
        else:
            st.error("Journey ended due to high risk!")
        
        if st.session_state.journey_choices:
            choices_df = pd.DataFrame(st.session_state.journey_choices)
            avg_risk = np.mean(choices_df["risk"])
            st.markdown(f"**Average Risk**: {avg_risk:.3f}")
            
            # Show each step
            for i, choice in enumerate(st.session_state.journey_choices):
                status = "✅ Survived" if choice["risk"] < st.session_state.journey_risk_threshold else "💥 Failed"
                st.write(f"Step {i+1}: {status} (Risk: {choice['risk']:.3f})")

        if st.button("🚀 New Journey", type="primary"):
            reset_journey()
        st.stop()

    # Current step
    current_step = st.session_state.journey_step
    st.header(f"Step {current_step} of {st.session_state.journey_total}")

    # Generate choices for current step
    if "journey_current_choices" not in st.session_state or len(st.session_state.journey_current_choices) != 3:
        st.session_state.journey_current_choices = generate_journey_choices(current_step)

    choices = st.session_state.journey_current_choices
    col1, col2, col3 = st.columns(3)

    # Choice A
    with col1:
        choice_a = choices[0]
        road_a = choice_a["road"]
        st.markdown(f"**Route A**")
        st.write(f"Type: {['Highway','Urban','Rural','Residential'][road_a['road_type']]}")
        st.write(f"Lanes: {road_a['num_lanes']}")
        st.write(f"Speed: {road_a['speed_limit']} km/h")
        st.write(f"Curvature: {road_a['curvature']:.1f}")
        st.write(f"Weather: {['Clear','Rain','Fog/Snow'][road_a['weather']]}")
        st.write(f"Lighting: {['Daylight','Twilight','Dark'][road_a['lighting']]}")

        if st.button("🛤️ Choose A", key=f"journey_a_{st.session_state.journey_id}_{current_step}"):
            risk = choice_a["risk"]
            survived = risk < st.session_state.journey_risk_threshold
            
            st.session_state.journey_choices.append({
                "step": current_step,
                "route": "A",
                "risk": risk,
                "survived": survived
            })
            
            if survived:
                st.session_state.journey_step += 1
                if st.session_state.journey_step > st.session_state.journey_total:
                    st.session_state.journey_active = False
                else:
                    st.session_state.journey_current_choices = generate_journey_choices(st.session_state.journey_step)
                st.success(f"Step {current_step}: Route A - Risk: {risk:.3f}")
            else:
                st.session_state.journey_active = False
                st.error(f"Step {current_step}: Route A - Risk: {risk:.3f} (too dangerous!)")
            
            st.rerun()

    # Choice B
    with col2:
        choice_b = choices[1]
        road_b = choice_b["road"]
        st.markdown(f"**Route B**")
        st.write(f"Type: {['Highway','Urban','Rural','Residential'][road_b['road_type']]}")
        st.write(f"Lanes: {road_b['num_lanes']}")
        st.write(f"Speed: {road_b['speed_limit']} km/h")
        st.write(f"Curvature: {road_b['curvature']:.1f}")
        st.write(f"Weather: {['Clear','Rain','Fog/Snow'][road_b['weather']]}")
        st.write(f"Lighting: {['Daylight','Twilight','Dark'][road_b['lighting']]}")

        if st.button("🛤️ Choose B", key=f"journey_b_{st.session_state.journey_id}_{current_step}"):
            risk = choice_b["risk"]
            survived = risk < st.session_state.journey_risk_threshold
            
            st.session_state.journey_choices.append({
                "step": current_step,
                "route": "B",
                "risk": risk,
                "survived": survived
            })
            
            if survived:
                st.session_state.journey_step += 1
                if st.session_state.journey_step > st.session_state.journey_total:
                    st.session_state.journey_active = False
                else:
                    st.session_state.journey_current_choices = generate_journey_choices(st.session_state.journey_step)
                st.success(f"Step {current_step}: Route B - Risk: {risk:.3f}")
            else:
                st.session_state.journey_active = False
                st.error(f"Step {current_step}: Route B - Risk: {risk:.3f} (too dangerous!)")
            
            st.rerun()

    # Choice C
    with col3:
        choice_c = choices[2]
        road_c = choice_c["road"]
        st.markdown(f"**Route C**")
        st.write(f"Type: {['Highway','Urban','Rural','Residential'][road_c['road_type']]}")
        st.write(f"Lanes: {road_c['num_lanes']}")
        st.write(f"Speed: {road_c['speed_limit']} km/h")
        st.write(f"Curvature: {road_c['curvature']:.1f}")
        st.write(f"Weather: {['Clear','Rain','Fog/Snow'][road_c['weather']]}")
        st.write(f"Lighting: {['Daylight','Twilight','Dark'][road_c['lighting']]}")

        if st.button("🛤️ Choose C", key=f"journey_c_{st.session_state.journey_id}_{current_step}"):
            risk = choice_c["risk"]
            survived = risk < st.session_state.journey_risk_threshold
            
            st.session_state.journey_choices.append({
                "step": current_step,
                "route": "C",
                "risk": risk,
                "survived": survived
            })
            
            if survived:
                st.session_state.journey_step += 1
                if st.session_state.journey_step > st.session_state.journey_total:
                    st.session_state.journey_active = False
                else:
                    st.session_state.journey_current_choices = generate_journey_choices(st.session_state.journey_step)
                st.success(f"Step {current_step}: Route C - Risk: {risk:.3f}")
            else:
                st.session_state.journey_active = False
                st.error(f"Step {current_step}: Route C - Risk: {risk:.3f} (too dangerous!)")
            
            st.rerun()

    # Progress bar
    steps_completed = len(st.session_state.journey_choices)
    progress = min(steps_completed, st.session_state.journey_total) / st.session_state.journey_total
    st.progress(progress)

    col1, col2 = st.columns(2)
    col1.metric("Steps Survived", steps_completed, st.session_state.journey_total)
    col2.metric("Risk Limit", f"{st.session_state.journey_risk_threshold}")
