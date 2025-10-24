import streamlit as st
import numpy as np
import pandas as pd
import joblib
from functools import lru_cache
from typing import Dict, Tuple

# Page config - FIRST
st.set_page_config(page_title="Road Safety App", page_icon="🛣️", layout="wide")

# Styling
st.markdown("""
<style>
.badge { padding: 6px 12px; border-radius: 20px; font-weight: bold; margin: 2px; display: inline-block; }
.stat { text-align: center; padding: 12px; background: #1a1a2e; border-radius: 8px; margin: 10px 0; }
.good { background: #2d5a27; color: #2ecc71; }
.warn { background: #4a4a1a; color: #f1c40f; }
.bad { background: #4a1a1a; color: #e74c3c; }
.step-card { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 10px; padding: 12px; margin: 8px 0; }
.kv { display: flex; gap: 8px; flex-wrap: wrap; font-size: 0.9rem; }
.kv span { background: #2a2a3e; padding: 4px 8px; border-radius: 6px; }
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
    st.success("✅ All models loaded!")
except Exception as e:
    st.error(f"❌ Models failed: {e}")
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

# ======================
# TABS
# ======================
tab_game, tab_explore, tab_journey = st.tabs(["🎮 Game", "🔎 Explore", "🚦 Journey"])

# ======================
# GAME TAB - FIXED
# ======================
with tab_game:
    # Game state with unique round tracking
    if "g_current_round" not in st.session_state:
        st.session_state.g_current_round = 1
        st.session_state.g_total_rounds = 10
        st.session_state.g_score = 0
        st.session_state.g_history = []
        st.session_state.g_game_active = True
        st.session_state.g_game_seed = np.random.randint(1000, 9999)

    def generate_game_pair(round_num):
        rng = np.random.default_rng(st.session_state.g_game_seed + round_num)
        return random_road(rng)

    def reset_game():
        st.session_state.g_current_round = 1
        st.session_state.g_score = 0
        st.session_state.g_history = []
        st.session_state.g_game_active = True
        st.session_state.g_game_seed = np.random.randint(1000, 9999)
        st.rerun()

    # Game summary (show only after completing rounds)
    if not st.session_state.g_game_active:
        st.header(f"🎯 Game Complete! Final Score: {st.session_state.g_score}/10")
        
        if st.session_state.g_history:
            df = pd.DataFrame(st.session_state.g_history)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Your Choices")
                for i, row in df.iterrows():
                    status = "✅ Correct" if row['correct'] else "❌ Wrong"
                    st.write(f"Round {row['round']}: Chose {row['choice']} (Risk: {row['risk_a']:.3f} vs {row['risk_b']:.3f}) - {status}")
            with col2:
                accuracy = np.mean([1 if x else 0 for x in df['correct']]) * 100
                st.subheader(f"Accuracy: {accuracy:.1f}%")
                st.bullet(["Good job!" if accuracy > 70 else "Room for improvement!", 
                          "Keep practicing!", "Nice work!"])
        
        if st.button("🔄 Play Again", type="primary"):
            reset_game()
        st.stop()

    # Current round
    current_round = st.session_state.g_current_round
    st.header(f"Round {current_round}/10")
    
    # Generate roads for this round
    if "g_roads" not in st.session_state or len(st.session_state.g_roads) != 2:
        st.session_state.g_roads = [
            generate_game_pair(current_round),
            generate_game_pair(current_round + 100)
        ]

    road_a, road_b = st.session_state.g_roads
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛣️ Road A")
        st.dataframe(pretty_table(road_a), use_container_width=True, height=300)
        if st.button("🚗 Choose A", key=f"game_choose_a_{current_round}"):
            risk_a = risk_only(road_a)
            risk_b = risk_only(road_b)
            correct = risk_a < risk_b
            st.session_state.g_history.append({
                'round': current_round,
                'choice': 'A',
                'risk_a': risk_a,
                'risk_b': risk_b,
                'correct': correct
            })
            
            if correct:
                st.session_state.g_score += 1
                st.success("✅ Good choice!")
            else:
                st.error("❌ Riskier than it looked!")
                st.write(f"Safe choice was B (A: {risk_a:.3f}, B: {risk_b:.3f})")
            
            st.session_state.g_current_round += 1
            if st.session_state.g_current_round > st.session_state.g_total_rounds:
                st.session_state.g_game_active = False
            else:
                st.session_state.g_roads = [
                    generate_game_pair(st.session_state.g_current_round),
                    generate_game_pair(st.session_state.g_current_round + 100)
                ]
            st.rerun()

    with col2:
        st.subheader("🛣️ Road B")
        st.dataframe(pretty_table(road_b), use_container_width=True, height=300)
        if st.button("🚗 Choose B", key=f"game_choose_b_{current_round}"):
            risk_a = risk_only(road_a)
            risk_b = risk_only(road_b)
            correct = risk_b < risk_a
            st.session_state.g_history.append({
                'round': current_round,
                'choice': 'B',
                'risk_a': risk_a,
                'risk_b': risk_b,
                'correct': correct
            })
            
            if correct:
                st.session_state.g_score += 1
                st.success("✅ Good choice!")
            else:
                st.error("❌ Riskier than it looked!")
                st.write(f"Safe choice was A (A: {risk_a:.3f}, B: {risk_b:.3f})")
            
            st.session_state.g_current_round += 1
            if st.session_state.g_current_round > st.session_state.g_total_rounds:
                st.session_state.g_game_active = False
            else:
                st.session_state.g_roads = [
                    generate_game_pair(st.session_state.g_current_round),
                    generate_game_pair(st.session_state.g_current_round + 100)
                ]
            st.rerun()

    # Progress bar
    progress = (st.session_state.g_current_round - 1) / st.session_state.g_total_rounds
    st.progress(progress)
    
    col_score, col_round = st.columns(2)
    col_score.metric("Score", st.session_state.g_score)
    col_round.metric("Round", f"{st.session_state.g_current_round-1}/{st.session_state.g_total_rounds}")

# ======================
# EXPLORE TAB
# ======================
with tab_explore:
    st.subheader("🔍 Explore Single Road")
    st.write("Configure parameters and see instant risk prediction")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        road_type = st.selectbox("Road Type", range(4), 
                                format_func=lambda x: ["Highway","Urban","Rural","Residential"][x],
                                key="explore_type")
        num_lanes = st.slider("Number of Lanes", 1, 6, 2, key="explore_lanes")
        curvature = st.slider("Curvature (0-2)", 0.0, 2.0, 0.5, 0.1, key="explore_curve")
        speed_limit = st.select_slider("Speed Limit (km/h)", [40,60,80,100,120], value=80, key="explore_speed")
    
    with col2:
        lighting = st.selectbox("Lighting", range(3), 
                               format_func=lambda x: ["Daylight","Twilight","Dark"][x],
                               key="explore_lighting")
        weather = st.selectbox("Weather", range(3), 
                              format_func=lambda x: ["Clear","Rain","Fog"][x],
                              key="explore_weather")
        time_of_day = st.selectbox("Time of Day", range(4),
                                  format_func=lambda x: ["Morning","Afternoon","Evening","Night"][x],
                                  key="explore_time")
        num_accidents = st.slider("Past Accidents (1 year)", 0, 100, 5, key="explore_accidents")

    col3, col4 = st.columns(2)
    with col3:
        road_signs = st.checkbox("Road Signs Present", True, key="explore_signs")
        public_road = st.checkbox("Public Road", True, key="explore_public")
    
    with col4:
        holiday = st.checkbox("Holiday Period", False, key="explore_holiday")
        school_season = st.checkbox("School Season", True, key="explore_school")

    # Build road config
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

    col_pred, col_table = st.columns([1, 3])
    with col_pred:
        if st.button("🎯 Calculate Risk", type="primary"):
            risk_score, model_parts = risk_and_parts(road_config)
            st.session_state.explore_risk = risk_score
            st.session_state.explore_parts = model_parts

    with col_table:
        if "explore_risk" in st.session_state:
            risk_score = st.session_state.explore_risk
            if risk_score < 0.3:
                st.success(f"**Risk Level:** Low ({risk_score:.3f})")
            elif risk_score < 0.6:
                st.warning(f"**Risk Level:** Medium ({risk_score:.3f})")
            else:
                st.error(f"**Risk Level:** High ({risk_score:.3f})")

    # Road summary
    st.subheader("Road Configuration")
    st.json({
        "Road Type": ["Highway","Urban","Rural","Residential"][road_type],
        "Lanes": num_lanes,
        "Curvature": f"{curvature:.1f}/2.0",
        "Speed Limit": f"{speed_limit} km/h",
        "Lighting": ["Daylight","Twilight","Dark"][lighting],
        "Weather": ["Clear","Rain","Fog"][weather],
        "Time of Day": ["Morning","Afternoon","Evening","Night"][time_of_day],
        "Road Signs": road_signs,
        "Public Road": public_road,
        "Holiday": holiday,
        "School Season": school_season,
        "Past Accidents": num_accidents
    })

# ======================
# JOURNEY TAB
# ======================
with tab_journey:
    st.subheader("🚦 Journey (Step-by-Step)")
    st.caption("Navigate multiple checkpoints. Choose the safest route at each step. Hidden risks until end!")

    # Journey state
    if "j_current_step" not in st.session_state:
        st.session_state.j_current_step = 1
        st.session_state.j_total_steps = 5
        st.session_state.j_threshold = 0.60
        st.session_state.j_path = []
        st.session_state.j_game_over = False
        st.session_state.j_journey_seed = np.random.randint(2000, 9999)

    def generate_journey_options(step):
        rng = np.random.default_rng(st.session_state.j_journey_seed + step)
        options = []
        for i in range(3):  # 3 options per step
            road = random_road(rng)
            risk = risk_only(road)
            options.append({"road": road, "risk": risk})
        return options

    def reset_journey():
        st.session_state.j_current_step = 1
        st.session_state.j_path = []
        st.session_state.j_game_over = False
        st.session_state.j_journey_seed = np.random.randint(2000, 9999)
        st.rerun()

    # Journey controls
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.j_total_steps = st.slider("Journey Length", 3, 8, 5, key="journey_length")
        st.session_state.j_threshold = st.slider("Risk Threshold", 0.4, 0.9, 0.60, 0.05, key="journey_threshold")
    with col2:
        if st.button("🚀 Start New Journey", type="primary"):
            reset_journey()

    if st.session_state.j_game_over:
        st.header("Journey Complete!")
        if len(st.session_state.j_path) == st.session_state.j_total_steps:
            st.success("✅ You reached the destination!")
        else:
            st.error("❌ Journey ended due to high risk!")
        
        st.write("Path taken:", st.session_state.j_path)
        if st.button("🔄 Try Again", type="primary"):
            reset_journey()
        st.stop()

    # Current step
    current_step = st.session_state.j_current_step
    st.header(f"Step {current_step} of {st.session_state.j_total_steps}")

    # Generate options for current step
    if "j_options" not in st.session_state or len(st.session_state.j_options) == 0:
        st.session_state.j_options = generate_journey_options(current_step)

    col1, col2, col3 = st.columns(3)
    options = st.session_state.j_options

    # Option 1
    with col1:
        opt1 = options[0]
        st.write(f"**Route A** - Risk: {'?' * 5}")
        st.write(f"Type: {['Highway','Urban','Rural','Residential'][opt1['road']['road_type']]}")
        st.write(f"Lanes: {opt1['road']['num_lanes']}")
        st.write(f"Speed: {opt1['road']['speed_limit']} km/h")
        st.write(f"Weather: {['Clear','Rain','Fog'][opt1['road']['weather']]}")
        st.write(f"Curvature: {opt1['road']['curvature']:.1f}")
        if st.button("🌟 Choose Route A", key=f"journey_a_{current_step}"):
            st.session_state.j_path.append({
                "step": current_step,
                "route": "A",
                "risk": opt1['risk'],
                "exceeded": opt1['risk'] >= st.session_state.j_threshold
            })
            
            if opt1['risk'] >= st.session_state.j_threshold:
                st.session_state.j_game_over = True
            else:
                st.session_state.j_current_step += 1
                if st.session_state.j_current_step > st.session_state.j_total_steps:
                    st.session_state.j_game_over = True
                else:
                    st.session_state.j_options = generate_journey_options(st.session_state.j_current_step)
            
            st.rerun()

    # Option 2
    with col2:
        opt2 = options[1]
        st.write(f"**Route B** - Risk: {'?' * 5}")
        st.write(f"Type: {['Highway','Urban','Rural','Residential'][opt2['road']['road_type']]}")
        st.write(f"Lanes: {opt2['road']['num_lanes']}")
        st.write(f"Speed: {opt2['road']['speed_limit']} km/h")
        st.write(f"Weather: {['Clear','Rain','Fog'][opt2['road']['weather']]}")
        st.write(f"Curvature: {opt2['road']['curvature']:.1f}")
        if st.button("🌟 Choose Route B", key=f"journey_b_{current_step}"):
            st.session_state.j_path.append({
                "step": current_step,
                "route": "B",
                "risk": opt2['risk'],
                "exceeded": opt2['risk'] >= st.session_state.j_threshold
            })
            
            if opt2['risk'] >= st.session_state.j_threshold:
                st.session_state.j_game_over = True
            else:
                st.session_state.j_current_step += 1
                if st.session_state.j_current_step > st.session_state.j_total_steps:
                    st.session_state.j_game_over = True
                else:
                    st.session_state.j_options = generate_journey_options(st.session_state.j_current_step)
            
            st.rerun()

    # Option 3
    with col3:
        opt3 = options[2]
        st.write(f"**Route C** - Risk: {'?' * 5}")
        st.write(f"Type: {['Highway','Urban','Rural','Residential'][opt3['road']['road_type']]}")
        st.write(f"Lanes: {opt3['road']['num_lanes']}")
        st.write(f"Speed: {opt3['road']['speed_limit']} km/h")
        st.write(f"Weather: {['Clear','Rain','Fog'][opt3['road']['weather']]}")
        st.write(f"Curvature: {opt3['road']['curvature']:.1f}")
        if st.button("🌟 Choose Route C", key=f"journey_c_{current_step}"):
            st.session_state.j_path.append({
                "step": current_step,
                "route": "C",
                "risk": opt3['risk'],
                "exceeded": opt3['risk'] >= st.session_state.j_threshold
            })
            
            if opt3['risk'] >= st.session_state.j_threshold:
                st.session_state.j_game_over = True
            else:
                st.session_state.j_current_step += 1
                if st.session_state.j_current_step > st.session_state.j_total_steps:
                    st.session_state.j_game_over = True
                else:
                    st.session_state.j_options = generate_journey_options(st.session_state.j_current_step)
            
            st.rerun()

    # Progress
    progress = (st.session_state.j_current_step - 1) / st.session_state.j_total_steps
    st.progress(progress)
    st.metric("Steps Completed", st.session_state.j_current_step - 1, st.session_state.j_total_steps)
