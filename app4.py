import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from functools import lru_cache
from typing import Dict, Tuple

# ======================
# App config (MUST BE FIRST streamlit command)
# ======================
# This was the main error: st.set_page_config was called twice.
# It must be called only ONCE, and it must be the FIRST st command.
st.set_page_config(
    page_title="Road Safety – Game | Explore | Journey", 
    page_icon="🛣️", 
    layout="wide"
)

# ======================
# DUMMY MODEL CREATOR (for testing)
# ======================
# Set this to False if you have your actual .pkl files.
# If True, this will create simple placeholder models so the app can run.
CREATE_DUMMY_MODELS = True

if CREATE_DUMMY_MODELS:
    try:
        # Check for sklearn, which is needed for the dummy models
        from sklearn.base import BaseEstimator
        
        class DummyModel(BaseEstimator):
            """A dummy model that predicts risk based on simple heuristics."""
            def predict(self, df):
                # Ensure DataFrame, not just dict
                if isinstance(df, dict):
                    df = pd.DataFrame([df])
                
                # A simple heuristic: risk increases with speed, curvature, and accidents
                # Use .get() for safety in case columns are missing in create_features
                speed = df.get('speed_limit', 80)
                curve = df.get('curvature', 0.5)
                acc = df.get('num_reported_accidents', 10)
                
                risk = (speed / 120.0) * 0.4 + (curve / 2.0) * 0.3 + (acc / 100.0) * 0.3
                return np.clip(risk.values, 0.05, 0.95)

        class DummyMetaModel(BaseEstimator):
            """A dummy meta-model that just averages the predictions."""
            def predict(self, X):
                # X is an array of shape (n_samples, 5)
                return np.mean(X, axis=1)

        models_needed = ["model_xgb.pkl", "model_lgb.pkl", "model_cat.pkl", "model_et.pkl", "model_rf.pkl", "meta.pkl"]
        for f in models_needed:
            if not os.path.exists(f):
                print(f"Creating dummy model: {f}") # Print to console
                if f == "meta.pkl":
                    joblib.dump(DummyMetaModel(), f)
                else:
                    joblib.dump(DummyModel(), f)
        
    except ImportError:
        st.error("Please install `scikit-learn` (`pip install scikit-learn`) to use the dummy model creator.")
        st.stop()
    except Exception as e:
        st.error(f"Error creating dummy models: {e}")
        st.stop()

# ======================
# File & Model Checks (Moved to UI)
# ======================
with st.expander("Show File & Model Status (Debug)"):
    st.write("Available files in directory:")
    file_list_code = ""
    for f in os.listdir('.'):
        if os.path.isfile(f):
            try:
                size_mb = os.path.getsize(f) / (1024*1024)
                file_list_code += f"- {f}: {size_mb:.2f}MB\n"
            except OSError:
                file_list_code += f"- {f}: (Could not get size)\n"
        else:
            file_list_code += f"- {f}: directory\n"
    st.code(file_list_code)
    
    models_needed = ["model_xgb.pkl", "model_lgb.pkl", "model_cat.pkl", "model_et.pkl", "model_rf.pkl", "meta.pkl"]
    missing = [m for m in models_needed if not os.path.exists(m)]
    if missing:
        st.error(f"Missing model files: {missing}. The app will stop.")
        st.info("If this is expected, set `CREATE_DUMMY_MODELS = True` at the top of the script to create placeholders.")
        st.stop()
    else:
        st.success("All required models found.")

# ======================
# App Styling
# ======================
st.markdown("""
<style>
:root {
  --bg:#0f1116; --card:#11141a; --stroke:#212737; --accent:#6C63FF; --text:#e8e8e8; --muted:#b9c0cc;
  --good:#2ecc71; --warn:#f1c40f; --bad:#e74c3c;
}
html, body, .block-container { background: var(--bg); color: var(--text); }
.card { background: var(--card); border:1px solid var(--stroke); border-radius:14px; padding:12px 14px; }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:0.85rem; }
.badge.good { background:rgba(46,204,113,0.12); color:var(--good); border:1px solid rgba(46,204,113,0.25); }
.badge.warn { background:rgba(241,196,15,0.12); color:var(--warn); border:1px solid rgba(241,196,15,0.25); }
.badge.bad  { background:rgba(231,76,60,0.12); color:var(--bad);  border:1px solid rgba(231,76,60,0.25); }
.ribbon { font-weight:800; font-size:1.2rem; color:#fff; padding:6px 10px; border-radius:10px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display:inline-block; }
.stat { background:#0f1320; border:1px solid #1f2740; border-radius:10px; padding:12px; text-align:center; }
.stat .v { font-size:1.6rem; font-weight:800; color:#fff; }
.btn-primary button { background:var(--accent)!important; border-color:var(--accent)!important; }
.good { background:#12351b; border-left:5px solid var(--good); padding:10px 12px; border-radius:10px; }
.warn { background:#3a2f12; border-left:5px solid var(--warn); padding:10px 12px; border-radius:10px; }
.bad  { background:#3a1616; border-left:5px solid var(--bad);  padding:10px 12px; border-radius:10px; }
.small { font-size:0.9rem; opacity:0.9; }

/* Micro-animations */
.fade-in { animation: fadeIn 260ms ease-out both; }
.slide-up { animation: slideUp 240ms ease-out both; }
@keyframes fadeIn { from { opacity:0 } to { opacity:1 } }
@keyframes slideUp { from { opacity:0; transform: translateY(6px) } to { opacity:1; transform: translateY(0) } }

/* Journey cards */
.step-card { background:var(--card); border:1px solid var(--stroke); border-radius:16px; padding:14px; transition: transform .12s ease; }
.step-card:hover { transform: translateY(-2px); }
.step-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.kv { display:flex; gap:6px; flex-wrap:wrap; color:var(--muted); font-size:0.9rem; }
.kv span { background:rgba(255,255,255,0.03); padding:4px 8px; border-radius:8px; border:1px solid rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)

st.title("🛣️ Road Safety – Game | Explore | Journey")

# ======================
# Helpers (no models)
# ======================
def random_road(rng: np.random.Generator) -> Dict:
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

def pretty_table(base: Dict) -> pd.DataFrame:
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
# Session state
# ======================
if "seed" not in st.session_state: st.session_state.seed = 42

# ======================
# Models + features
# ======================
@st.cache_resource(show_spinner="Loading predictive models...")
def load_models():
    xgb = joblib.load("model_xgb.pkl")
    lgb = joblib.load("model_lgb.pkl")
    cat = joblib.load("model_cat.pkl")
    et  = joblib.load("model_et.pkl")
    rf  = joblib.load("model_rf.pkl")
    meta = joblib.load("meta.pkl")
    return xgb, lgb, cat, et, rf, meta

BASE_COLS = [
    "road_type","num_lanes","curvature","speed_limit","lighting","weather",
    "road_signs_present","public_road","time_of_day","holiday","school_season",
    "num_reported_accidents"
]

def create_features(base: Dict) -> pd.DataFrame:
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

try:
    xgb_model, lgb_model, cat_model, et_model, rf_model, meta_model = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.info("This might be because the dummy models are incompatible with your `create_features` function, or the real models are corrupt.")
    st.stop()


@lru_cache(maxsize=2048)
def risk_and_parts_cached(road_tuple: Tuple) -> Tuple[float, Tuple[float,float,float,float,float]]:
    base = {k: v for k, v in zip(BASE_COLS, road_tuple)}
    feats = create_features(base)
    # Use predict_proba if available (common for classifiers), otherwise predict
    def get_pred(model, data):
        if hasattr(model, "predict_proba"):
            # Assuming risk is the probability of class 1
            return model.predict_proba(data)[:, 1][0]
        else:
            return model.predict(data)[0]

    p1 = float(get_pred(xgb_model, feats))
    p2 = float(get_pred(lgb_model, feats))
    p3 = float(get_pred(cat_model, feats))
    p4 = float(get_pred(et_model, feats))
    p5 = float(get_pred(rf_model, feats))
    
    meta_input = np.array([[p1,p2,p3,p4,p5]])
    final = float(np.clip(meta_model.predict(meta_input)[0], 0, 1))
    
    return final, (p1,p2,p3,p4,p5)

def risk_only(base: Dict) -> float:
    t = tuple(base[k] for k in BASE_COLS)
    y, _ = risk_and_parts_cached(t)
    return y

def risk_and_parts(base: Dict) -> Tuple[float, Dict[str,float]]:
    t = tuple(base[k] for k in BASE_COLS)
    y, parts = risk_and_parts_cached(t)
    return y, {"XGBoost":parts[0],"LightGBM":parts[1],"CatBoost":parts[2],"ExtraTrees":parts[3],"RandomForest":parts[4]}

# ======================
# Tabs: Game | Explore | Journey
# ======================
tab_game, tab_explore, tab_journey = st.tabs([
    "🎮 Game – Pick the safer road (10 rounds)",
    "🔎 Explore a single road",
    "🚦 Journey (start → finish)"
])

# ---------------------- GAME (best-of-10 with final summary) ----------------------
with tab_game:
    # State
    if "g_round" not in st.session_state: st.session_state.g_round = 1
    if "g_max" not in st.session_state: st.session_state.g_max = 10
    if "g_score" not in st.session_state: st.session_state.g_score = 0
    if "g_pair" not in st.session_state:
        rng = np.random.default_rng(777)
        st.session_state.g_pair = (random_road(rng), random_road(rng))
    if "g_last" not in st.session_state: st.session_state.g_last = None              # (correct, rA, rB)
    if "g_hist" not in st.session_state: st.session_state.g_hist = []               # [{round, choice, safer, rA, rB, correct, chosen_base, other_base}]
    if "g_finished" not in st.session_state: st.session_state.g_finished = False

    def summarize_mistakes(rows):
        counts = {"night":0,"fog":0,"high_speed":0,"sharp_curve":0,"few_lanes":0,"no_signs":0,"acc_hotspot":0}
        for h in rows:
            if h["correct"]: 
                continue
            b = h["chosen_base"]
            if b["lighting"] == 2: counts["night"] += 1
            if b["weather"] >= 1: counts["fog"] += 1
            if b["speed_limit"] > 80: counts["high_speed"] += 1
            if b["curvature"] > 1.0: counts["sharp_curve"] += 1
            if b["num_lanes"] <= 2: counts["few_lanes"] += 1
            if not b["road_signs_present"]: counts["no_signs"] += 1
            if b["num_reported_accidents"] >= 50: counts["acc_hotspot"] += 1
        labels = {
            "night":"Night driving", "fog":"Poor weather", "high_speed":"High speed",
            "sharp_curve":"Sharp curvature", "few_lanes":"Narrow road (≤2 lanes)",
            "no_signs":"Missing road signs", "acc_hotspot":"Accident hotspot"
        }
        ranked = sorted(counts.items(), key=lambda x:x[1], reverse=True)
        return [(labels[k], v) for k,v in ranked if v>0][:3]

    # Final summary
    def game_summary():
        total = len(st.session_state.g_hist)
        if total == 0:
            st.info("No rounds played.")
            return
        correct = sum(1 for h in st.session_state.g_hist if h["correct"])
        acc = 100.0 * correct / total
        st.subheader("Game summary")
        st.success(f"Score: {correct} / {total}  ({acc:.1f}%)")

        top_mistakes = summarize_mistakes(st.session_state.g_hist)
        cL, cR = st.columns(2)
        with cL:
            st.markdown("Where you went wrong")
            if top_mistakes:
                for name, cnt in top_mistakes:
                    st.markdown(f"- {name}: {cnt} rounds")
            else:
                st.markdown("- No consistent mistakes detected.")
        with cR:
            st.markdown("Improvement tips")
            st.markdown(
                "- Prefer good lighting and clear signage.\n"
                "- Reduce speed at night, on curves, or in bad weather.\n"
                "- Avoid known accident hotspots; detour if necessary.\n"
                "- More lanes are generally safer at the same speed."
            )

        rows = []
        for h in st.session_state.g_hist:
            rows.append({
                "Round": h["round"],
                "Choice": h["choice"],
                "Safer": h["safer"],
                "Risk A": f"{h['rA']:.3f}",
                "Risk B": f"{h['rB']:.3f}",
                "Correct": "Yes" if h["correct"] else "No"
            })
        st.markdown("Round-by-round")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Play again", type="primary"):
                st.session_state.g_round = 1
                st.session_state.g_score = 0
                st.session_state.g_last = None
                st.session_state.g_hist = []
                st.session_state.g_finished = False
                rng = np.random.default_rng(777)
                st.session_state.g_pair = (random_road(rng), random_road(rng))
                st.rerun()

    # If finished, show summary and stop drawing the round UI
    if st.session_state.g_finished:
        game_summary()
        st.stop()

    # Round UI
    st.subheader(f"Round {st.session_state.g_round} / {st.session_state.g_max}")
    A, B = st.session_state.g_pair
    colA, colB = st.columns(2)

    with colA:
        st.markdown('<span class="badge warn">Road A</span>', unsafe_allow_html=True)
        st.dataframe(pretty_table(A), hide_index=True, use_container_width=True, height=300)
        if st.button("Choose Road A", key=f"g_pick_A_{st.session_state.g_round}", use_container_width=True, type="primary"):            
            rA, rB = risk_only(A), risk_only(B)
            safer = "A" if rA < rB else "B"
            correct = (safer == "A")
            st.session_state.g_last = (correct, rA, rB)
            if correct: st.session_state.g_score += 1
            st.session_state.g_hist.append({
                "round": st.session_state.g_round, "choice":"A", "safer":safer,
                "rA":rA, "rB":rB, "correct":correct, "chosen_base":A, "other_base":B
            })
            st.session_state.g_round += 1
            if st.session_state.g_round > st.session_state.g_max:
                st.session_state.g_finished = True
            else:
                rng = np.random.default_rng(st.session_state.g_round + 999)
                st.session_state.g_pair = (random_road(rng), random_road(rng))
                st.rerun()

    with colB:
        st.markdown('<span class="badge warn">Road B</span>', unsafe_allow_html=True)
        st.dataframe(pretty_table(B), hide_index=True, use_container_width=True, height=300)
        if st.button("Choose Road B", key=f"g_pick_B_{st.session_state.g_round}", use_container_width=True, type="primary"):            
            rA, rB = risk_only(A), risk_only(B)
            safer = "A" if rA < rB else "B"
            correct = (safer == "B")
            st.session_state.g_last = (correct, rA, rB)
            if correct: st.session_state.g_score += 1
            st.session_state.g_hist.append({
                "round": st.session_state.g_round, "choice":"B", "safer":safer,
                "rA":rA, "rB":rB, "correct":correct, "chosen_base":B, "other_base":A
            })
            st.session_state.g_round += 1
            if st.session_state.g_round > st.session_state.g_max:
                st.session_state.g_finished = True
            else:
                rng = np.random.default_rng(st.session_state.g_round + 999)
                st.session_state.g_pair = (random_road(rng), random_road(rng))
                st.rerun()

    # Per-round feedback
    if st.session_state.g_last and not st.session_state.g_finished:
        correct, rA, rB = st.session_state.g_last
        if correct:
            st.success(f"Correct. Risk(A)={rA:.3f} • Risk(B)={rB:.3f}")
        else:
            st.error(f"Not correct. Risk(A)={rA:.3f} • Risk(B)={rB:.3f}")

    # Scoreboard
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f'<div class="stat">Score<br><span class="v">{st.session_state.g_score}</span></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat">Round<br><span class="v">{min(st.session_state.g_round, st.session_state.g_max)}/{st.session_state.g_max}</span></div>', unsafe_allow_html=True)


# ---------------------- EXPLORE (fast) ----------------------
with tab_explore:
    st.subheader("Configure a road and predict risk")
    road_names  = ["Highway","Urban","Rural","Residential"]
    light_names = ["Daylight","Twilight","Dark"]
    weather_names = ["Clear","Rain","Fog/Snow"]
    tod_names = ["Morning","Afternoon","Evening","Night"]

    cA, cB, cC, cD = st.columns(4)
    with cA:
        road_type = st.selectbox("Road Type", list(range(4)), format_func=lambda i: road_names[i])
        num_lanes = st.slider("Lanes", 1, 6, 2)
    with cB:
        curvature = st.slider("Curvature", 0.0, 2.0, 0.5, 0.1)
        speed_limit = st.select_slider("Speed Limit (km/h)", options=[40,60,80,100,120], value=80)
    with cC:
        lighting = st.selectbox("Lighting", list(range(3)), format_func=lambda i: light_names[i])
        weather  = st.selectbox("Weather", list(range(3)), format_func=lambda i: weather_names[i])
    with cD:
        time_of_day = st.selectbox("Time of Day", list(range(4)), format_func=lambda i: tod_names[i])
        num_acc = st.slider("Past Accidents (1y)", 0, 100, 5)
    colE1, colE2, colE3 = st.columns(3)
    with colE1: road_signs_present = st.checkbox("Road Signs Present", True)
    with colE2: public_road = st.checkbox("Public Road", True)
    with colE3:
        holiday = st.checkbox("Holiday", False)
        school_season = st.checkbox("School Season", True)

    base = {
        "road_type": road_type, "num_lanes": num_lanes, "curvature": curvature,
        "speed_limit": speed_limit, "lighting": lighting, "weather": weather,
        "road_signs_present": road_signs_present, "public_road": public_road,
        "time_of_day": time_of_day, "holiday": holiday,
        "school_season": school_season, "num_reported_accidents": num_acc
    }

    if st.button("🎯 Predict risk", type="primary"):
        score, _ = risk_and_parts(base)
        tag = "good" if score<0.3 else ("warn" if score<0.6 else "bad")
        st.markdown(f'<div class="{tag} slide-up"><b>Risk</b>: {score:.4f}</div>', unsafe_allow_html=True)
    st.dataframe(pretty_table(base), hide_index=True, use_container_width=True)


# ---------------------- JOURNEY (start → finish, complete) ----------------------
with tab_journey:
    # Safe init for all keys used by Journey
    if "j_step" not in st.session_state: st.session_state.j_step = 1
    if "j_max"  not in st.session_state: st.session_state.j_max = 5
    if "j_thr"  not in st.session_state: st.session_state.j_thr = 0.60
    if "j_seed" not in st.session_state: st.session_state.j_seed = 2025
    if "j_hist" not in st.session_state: st.session_state.j_hist = []   # [{step, chosen_risk, best_risk, optimal, chosen_base}]
    if "j_alive" not in st.session_state: st.session_state.j_alive = True
    if "j_opts" not in st.session_state: st.session_state.j_opts = None  # current step options
    if "j_opt_count" not in st.session_state: st.session_state.j_opt_count = 3
    if "j_all_opts" not in st.session_state: st.session_state.j_all_opts = {}  # step -> options shown

    st.subheader("Start → Finish (no‑spoilers mode)")
    st.markdown("<p class='small'>Pick the safest option at each checkpoint. Risks are hidden until the end. If your chosen option’s risk ≥ threshold, the journey ends immediately.</p>", unsafe_allow_html=True)

    # Difficulty controls
    d1, d2, d3, d4 = st.columns([1,1,1,2])
    with d1:
        mode = st.selectbox("Mode", ["Easy","Normal","Hard","Custom"], index=1)
    with d2:
        if mode=="Easy":
            st.session_state.j_thr = 0.65
            st.session_state.j_max = 4
        elif mode=="Normal":
            st.session_state.j_thr = 0.60
            st.session_state.j_max = 5
        elif mode=="Hard":
            st.session_state.j_thr = 0.55
            st.session_state.j_max = 6
        st.caption(f"Threshold: {st.session_state.j_thr:.2f} • Steps: {st.session_state.j_max}")
    with d3:
        st.session_state.j_opt_count = st.select_slider("Options/step", [2,3], value=st.session_state.j_opt_count, key="j_opt_per_step")
    with d4:
        if st.button("🔄 Restart journey", type="primary"):
            st.session_state.j_step = 1
            st.session_state.j_hist = []
            st.session_state.j_alive = True
            st.session_state.j_seed += 17
            st.session_state.j_opts = None
            st.session_state.j_all_opts = {}
            st.rerun()

    st.markdown("---")

    # Milestone banner (cosmetic)
    if 1 <= st.session_state.j_step <= st.session_state.j_max:
        mile = {1:"Departure", 2:"City Outskirts", 3:"Hills", 4:"Highway Merge", 5:"Suburbs", 6:"Downtown"}.get(st.session_state.j_step, f"Stage {st.session_state.j_step}")
        st.markdown(f'<span class="ribbon fade-in">🧭 {mile}</span>', unsafe_allow_html=True)

    # Recap tray (risks hidden until the end)
    if st.session_state.j_hist:
        with st.expander("📜 Journey recap so far", expanded=False):
            finished = (not st.session_state.j_alive) or (st.session_state.j_step>st.session_state.j_max)
            rows = []
            for h in st.session_state.j_hist:
                rows.append({
                    "Step": h["step"],
                    "Chosen": f'{h["chosen_risk"]:.3f}' if finished else "hidden",
                    "BestAtStep": f'{h["best_risk"]:.3f}' if finished else "hidden",
                    "Optimal?": ("✅" if h["optimal"] else "❌") if finished else "—"
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Helpers
    def gen_options(step, n, seed):
        rng = np.random.default_rng(seed + step*911)
        opts = []
        for k in range(n):
            b = random_road(rng)
            y = risk_only(b)   # cached
            opts.append({"base":b, "risk":float(y)})
        opts.sort(key=lambda d: d["risk"])
        return opts

    def quick_tips_for(base: Dict):
        tips = []
        if base["lighting"]==2: tips.append("Choose options with better lighting when possible.")
        if base["weather"]>=1: tips.append("Prefer clear weather paths or slow down significantly.")
        if base["speed_limit"]>80: tips.append("Favor lower speed limits when curves/visibility are bad.")
        if base["num_lanes"]<=2: tips.append("Wider roads (more lanes) are safer at similar speeds.")
        if not base["road_signs_present"]: tips.append("Presence of signage helps; pick those routes.")
        if base["num_reported_accidents"]>=50: tips.append("Avoid accident hotspots; detour if risks are high.")
        return tips[:3]

    def show_summary(alive: bool):
        hist = st.session_state.j_hist
        if not hist:
            st.info("No steps taken.")
            return
        avg_risk = float(np.mean([h["chosen_risk"] for h in hist]))
        if alive and len(hist)==st.session_state.j_max:
            st.balloons()
            st.markdown(f'<div class="good slide-up"><b>🏁 Destination reached</b> • average chosen risk {avg_risk:.3f}</div>', unsafe_allow_html=True)
        else:
            bad = [h for h in hist if h["chosen_risk"] >= st.session_state.j_thr]
            if bad:
                st.markdown(f'<div class="bad slide-up">💥 Failed at step {bad[0]["step"]}: chosen {bad[0]["chosen_risk"]:.3f} ≥ threshold {st.session_state.j_thr:.2f}</div>', unsafe_allow_html=True)
                st.markdown("#### 🎯 Quick coaching")
                for t in quick_tips_for(bad[0]["chosen_base"]): st.markdown(f"- {t}")

        # Your path (now reveal numbers)
        rows = [{"Step":h["step"], "Chosen":f'{h["chosen_risk"]:.3f}', "BestAtStep":f'{h["best_risk"]:.3f}', "Delta": f'{(h["chosen_risk"]-h["best_risk"]):+.3f}', "Optimal?":"✅" if h["optimal"] else "❌"} for h in hist]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Optimal path reconstruction using stored options
        st.markdown("### 🧭 Optimal path (model’s picks at each shown step)")
        best_steps = []
        for step in range(1, len(hist)+1):
            opts = st.session_state.j_all_opts.get(step)
            if not opts: continue
            best_opt = min(opts, key=lambda d: d["risk"])
            best_steps.append({"Step": step, "BestRisk": best_opt["risk"], "Base": best_opt["base"]})

        if best_steps:
            opt_avg = float(np.mean([s["BestRisk"] for s in best_steps]))
            st.markdown(f'<div class="warn slide-up"><b>Optimal average risk</b>: {opt_avg:.3f}</div>', unsafe_allow_html=True)
            
            # === CORRECTED/CLEANED CODE BLOCK ===
            # The original script had a broken, half-finished loop here.
            # This is the corrected, functional version.
            tbl = []
            for s in best_steps:
                b = s["Base"]
                desc = " ".join([
                    ["Hwy","Urban","Rural","Res"][b["road_type"]],
                    f'{b["num_lanes"]}L',
                    f'curv {b["curvature"]:.1f}',
                    f'{b["speed_limit"]}km/h',
                    ["Day","Twilight","Dark"][b["lighting"]],
                    ["Clear","Rain","Fog"][b["weather"]], # Note: this assumes weather 0, 1, 2
                    "signs" if b["road_signs_present"] else "no-signs",
                    f'{b["num_reported_accidents"]}acc'
                ])
                tbl.append({"Step": s["Step"], "BestRisk": f'{s["BestRisk"]:.3f}', "Option": desc})
            st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)
            # === END OF CORRECTION ===
            
        else:
            st.write("Optimal path could not be reconstructed (no options tracked).")

    # If journey ended (dead or finished), show summary and stop
    if (not st.session_state.j_alive) or (st.session_state.j_step>st.session_state.j_max):
        show_summary(st.session_state.j_alive and (st.session_state.j_step>st.session_state.j_max))
        st.stop()

    # Prepare current step options
    if st.session_state.j_opts is None:
        st.session_state.j_opts = gen_options(st.session_state.j_step, st.session_state.j_opt_count, st.session_state.j_seed)
        st.session_state.j_all_opts[st.session_state.j_step] = st.session_state.j_opts

    st.markdown(f"### Checkpoint {st.session_state.j_step}/{st.session_state.j_max}")

    # Render options without risk values (hidden)
    cols = st.columns(st.session_state.j_opt_count)
    for i, col in enumerate(cols):
        with col:
            # Handle potential case where options weren't generated
            if i >= len(st.session_state.j_opts):
                continue
                
            opt = st.session_state.j_opts[i]; b = opt["base"]; r = opt["risk"]
            st.markdown(
                f'<div class="step-card fade-in">'
                f'<div class="step-header"><div class="badge warn">Option {i+1}</div>'
                f'<div>Checkpoint {st.session_state.j_step}</div></div>',
                unsafe_allow_html=True
            )
            chips = [
                f'🚗 {["Hwy","Urban","Rural","Res"][b["road_type"]]}',
                f'🛤️ {b["num_lanes"]} lanes',
                f'🌀 {b["curvature"]:.1f}',
                f'🚧 {b["speed_limit"]} km/h',
                f'💡 {["Day","Twilight","Dark"][b["lighting"]]}',
                f'🌦️ {["Clear","Rain","Fog"][b["weather"]]}', # Note: this assumes weather 0, 1, 2
                f'🚸 {"Signs" if b["road_signs_present"] else "No signs"}',
                f'📊 {b["num_reported_accidents"]} acc'
            ]
            st.markdown('<div class="kv">' + "".join([f"<span>{x}</span>" for x in chips]) + "</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button(f"✅ Choose Option {i+1}", key=f"j_pick_{st.session_state.j_step}_{i}", use_container_width=True):
                best = min([o["risk"] for o in st.session_state.j_opts])
                optimal = (abs(r-best) < 1e-9) or (r==best)
                st.session_state.j_hist.append({
                    "step": st.session_state.j_step, "chosen_risk": r, "best_risk": best,
                    "optimal": optimal, "chosen_base": b
                })
                if r >= st.session_state.j_thr:
                    st.session_state.j_alive = False
                    show_summary(False)
                    st.stop()
                st.session_state.j_step += 1
                st.session_state.j_opts = None
                if st.session_state.j_step > st.session_state.j_max:
                    show_summary(True)
                    st.stop()
                st.rerun()