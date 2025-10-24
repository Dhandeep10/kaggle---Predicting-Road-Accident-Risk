import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from functools import lru_cache
from typing import Dict, Tuple

# ======================
# App config
# ======================
st.set_page_config(page_title="Road Safety – Game + Explorer", page_icon="🛣️", layout="wide")
MAX_ROUNDS = 10

# ======================
# Simple styling
# ======================
st.markdown("""
<style>
.card { background:#0f1116; border:1px solid #212737; border-radius:12px; padding:12px 14px; }
.stat { background:#0f1320; border:1px solid #1f2740; border-radius:10px; padding:12px; text-align:center; }
.stat .v { font-size:1.6rem; font-weight:800; color:#fff; }
.btn-primary button { background:#6C63FF!important; border-color:#6C63FF!important; }
.good { background:#12351b; border-left:5px solid #2ecc71; padding:10px 12px; border-radius:10px; }
.warn { background:#3a2f12; border-left:5px solid #f1c40f; padding:10px 12px; border-radius:10px; }
.bad  { background:#3a1616; border-left:5px solid #e74c3c; padding:10px 12px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.title("🛣️ Road Safety – Pick the Safer Road + Explore Conditions")

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

def summarize_weakness(history):
    counts = { "night":0,"fog":0,"high_speed":0,"sharp_curve":0,"few_lanes":0,"no_signs":0,"accident_hotspot":0 }
    wrong = [h for h in history if not h["correct"]]
    for h in wrong:
        road = h["roadA"] if h["chosen"]=="A" else h["roadB"]
        if road["lighting"] == 2: counts["night"] += 1
        if road["weather"] >= 1: counts["fog"] += 1
        if road["speed_limit"] > 80: counts["high_speed"] += 1
        if road["curvature"] > 1.0: counts["sharp_curve"] += 1
        if road["num_lanes"] <= 2: counts["few_lanes"] += 1
        if not road["road_signs_present"]: counts["no_signs"] += 1
        if road["num_reported_accidents"] >= 50: counts["accident_hotspot"] += 1
    labels = {
        "night":"Night driving","fog":"Poor weather","high_speed":"High speed",
        "sharp_curve":"Sharp curvature","few_lanes":"Narrow road (≤2 lanes)",
        "no_signs":"Missing road signs","accident_hotspot":"Accident hotspot"
    }
    ranked = [(labels[k], v) for k,v in sorted(counts.items(), key=lambda x:x[1], reverse=True) if v>0][:3]
    return ranked

# ======================
# Session state
# ======================
if "seed" not in st.session_state: st.session_state.seed = 42
if "score" not in st.session_state: st.session_state.score = 0
if "rounds" not in st.session_state: st.session_state.rounds = 0
if "pair" not in st.session_state:
    rng = np.random.default_rng(st.session_state.seed)
    st.session_state.pair = (random_road(rng), random_road(rng))
if "last_result" not in st.session_state: st.session_state.last_result = None
if "history" not in st.session_state: st.session_state.history = []
if "show_gameover" not in st.session_state: st.session_state.show_gameover = False

def new_pair():
    rng = np.random.default_rng()
    st.session_state.pair = (random_road(rng), random_road(rng))

# Keep flag if already done
if st.session_state.rounds >= MAX_ROUNDS:
    st.session_state.show_gameover = True

# ======================
# Models + features
# ======================
@st.cache_resource(show_spinner=True)
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

xgb_model, lgb_model, cat_model, et_model, rf_model, meta_model = load_models()

@lru_cache(maxsize=2048)
def risk_from_tuple(road_tuple: Tuple) -> float:
    base = {k: v for k, v in zip(BASE_COLS, road_tuple)}
    feats = create_features(base)
    parts = [
        float(xgb_model.predict(feats)[0]),
        float(lgb_model.predict(feats)[0]),
        float(cat_model.predict(feats)[0]),
        float(et_model.predict(feats)[0]),
        float(rf_model.predict(feats)[0]),
    ]
    final = float(np.clip(meta_model.predict(np.array([parts]))[0], 0, 1))
    return final

def risk_and_parts(base: Dict) -> Tuple[float, Dict[str, float]]:
    feats = create_features(base)
    p = {
        "XGBoost": float(xgb_model.predict(feats)[0]),
        "LightGBM": float(lgb_model.predict(feats)[0]),
        "CatBoost": float(cat_model.predict(feats)[0]),
        "ExtraTrees": float(et_model.predict(feats)[0]),
        "RandomForest": float(rf_model.predict(feats)[0]),
    }
    final = float(np.clip(meta_model.predict(np.array([list(p.values())]))[0], 0, 1))
    return final, p

# ======================
# If game finished, render clickable Game Over section and stop
# ======================
if st.session_state.show_gameover:
    st.markdown("## 🏁 Game Over")
    total = len(st.session_state.history)
    correct = sum(1 for h in st.session_state.history if h["correct"])
    pct = 100.0 * correct / total if total else 0.0
    st.write(f"Score: **{correct} / {MAX_ROUNDS}**  ({pct:.1f}%)")

    top_mistakes = summarize_weakness(st.session_state.history)
    colL, colR = st.columns(2)
    with colL:
        st.markdown("#### ❌ Where you went wrong")
        if top_mistakes:
            for label, count in top_mistakes:
                st.markdown(f"- {label}: **{count}** rounds")
        else:
            st.markdown("- Nice! No consistent mistakes spotted.")
    with colR:
        st.markdown("#### 🛡️ Safety guidance")
        st.markdown("""
- Prefer roads with clear signage and good lighting.
- Reduce speed on curves, at night, or in bad weather.
- Avoid known accident hotspots or choose alternate routes.
- Wider roads (more lanes) generally reduce risk at the same speed.
- Worst combo: high speed + sharp curve + poor visibility — slow down.
        """)

    with st.expander("🔎 See round-by-round outcomes"):
        rows = []
        for h in st.session_state.history:
            rows.append({
                "Round": h["round"], "Chosen": h["chosen"], "Safer": h["safer"],
                "Risk A": f'{h["riskA"]:.4f}', "Risk B": f'{h["riskB"]:.4f}',
                "Correct": "✅" if h["correct"] else "❌"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", data=df.to_csv(index=False),
                           file_name="road_safety_results.csv", mime="text/csv")

    if st.button("🔁 Play again", type="primary"):
        st.session_state.score = 0
        st.session_state.rounds = 0
        st.session_state.last_result = None
        st.session_state.history = []
        st.session_state.show_gameover = False
        new_pair()
        st.rerun()

    st.stop()

# ======================
# Tabs
# ======================
tab_game, tab_explore = st.tabs(["🎮 Game – Pick the safer road (10 rounds)", "🔎 Explore a single road"])

# ---------------------- Game tab ----------------------
with tab_game:
    st.subheader(f"Round {st.session_state.rounds + 1} of {MAX_ROUNDS}")

    A, B = st.session_state.pair
    swap = bool(np.random.default_rng().integers(0, 2))
    left, right = (A, B) if not swap else (B, A)
    left_label, right_label = ("A","B") if not swap else ("B","A")

    c1, _, c3 = st.columns([1.6, 0.2, 1.6])
    with c1:
        st.markdown(f"#### Road {left_label}")
        st.dataframe(pretty_table(left), use_container_width=True, hide_index=True)
    with c3:
        st.markdown(f"#### Road {right_label}")
        st.dataframe(pretty_table(right), use_container_width=True, hide_index=True)

    st.markdown("---")
    colL, _, colR = st.columns([1, 0.2, 1])
    choice = None
    with colL:
        if st.button(f"✅ Road {left_label} is safer", use_container_width=True, type="primary"):
            choice = left_label
    with colR:
        if st.button(f"✅ Road {right_label} is safer", use_container_width=True, type="primary"):
            choice = right_label

    if choice and st.session_state.rounds < MAX_ROUNDS:
        tupA = tuple(A[k] for k in BASE_COLS)
        tupB = tuple(B[k] for k in BASE_COLS)
        riskA = risk_from_tuple(tupA)
        riskB = risk_from_tuple(tupB)
        safer = "A" if riskA < riskB else "B"
        chosen_true = choice if not swap else ("A" if choice=="B" else "B")
        correct = (chosen_true == safer)

        st.session_state.rounds += 1
        if correct: st.session_state.score += 1
        st.session_state.last_result = ("correct" if correct else "wrong", safer, riskA, riskB)

        st.session_state.history.append({
            "round": st.session_state.rounds,
            "chosen": chosen_true, "safer": safer,
            "riskA": riskA, "riskB": riskB,
            "roadA": A, "roadB": B, "correct": correct
        })

        if st.session_state.rounds < MAX_ROUNDS:
            new_pair()
            st.rerun()
        else:
            st.session_state.show_gameover = True  # render at top next cycle

    if st.session_state.last_result and st.session_state.rounds < MAX_ROUNDS:
        status, safer, rA, rB = st.session_state.last_result
        if status == "correct":
            st.success(f"🎉 Correct! Safer road: {safer}.  Risk(A)={rA:.4f} | Risk(B)={rB:.4f}")
        else:
            st.error(f"❌ Not quite. Safer road: {safer}.  Risk(A)={rA:.4f} | Risk(B)={rB:.4f}")

    s1, s2, _, _ = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat">Score<br><span class="v">{st.session_state.score}</span></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat">Round<br><span class="v">{st.session_state.rounds}/{MAX_ROUNDS}</span></div>', unsafe_allow_html=True)

    if st.session_state.rounds < MAX_ROUNDS:
        if st.button("➡️ Next", type="secondary"):
            new_pair()
            st.rerun()

# ---------------------- Explore tab ----------------------
with tab_explore:
    st.subheader("Configure a road and predict risk")

    road_names  = ["Highway","Urban","Rural","Residential"]
    light_names = ["Daylight","Twilight","Dark"]
    weather_names = ["Clear","Rain","Fog/Snow"]
    tod_names = ["Morning","Afternoon","Evening","Night"]

    colA, colB, colC, colD = st.columns(4)
    with colA:
        road_type = st.selectbox("Road Type", list(range(4)), format_func=lambda i: road_names[i])
        num_lanes = st.slider("Lanes", 1, 6, 2)
    with colB:
        curvature = st.slider("Curvature", 0.0, 2.0, 0.5, 0.1)
        speed_limit = st.select_slider("Speed Limit (km/h)", options=[40,60,80,100,120], value=80)
    with colC:
        lighting = st.selectbox("Lighting", list(range(3)), format_func=lambda i: light_names[i])
        weather  = st.selectbox("Weather", list(range(3)), format_func=lambda i: weather_names[i])
    with colD:
        time_of_day = st.selectbox("Time of Day", list(range(4)), format_func=lambda i: tod_names[i])
        num_acc = st.slider("Past Accidents (1y)", 0, 100, 5)

    cE1, cE2, cE3 = st.columns(3)
    with cE1: road_signs_present = st.checkbox("Road Signs Present", True)
    with cE2: public_road = st.checkbox("Public Road", True)
    with cE3:
        holiday = st.checkbox("Holiday", False)
        school_season = st.checkbox("School Season", True)

    base = {
        "road_type": road_type, "num_lanes": num_lanes, "curvature": curvature,
        "speed_limit": speed_limit, "lighting": lighting, "weather": weather,
        "road_signs_present": road_signs_present, "public_road": public_road,
        "time_of_day": time_of_day, "holiday": holiday,
        "school_season": school_season, "num_reported_accidents": num_acc
    }

    if st.button("🎯 Compute risk", type="primary"):
        score, parts = risk_and_parts(base)
        if score < 0.3:
            st.markdown(f'<div class="good"><b>LOW RISK</b> — score: {score:.4f}</div>', unsafe_allow_html=True)
        elif score < 0.6:
            st.markdown(f'<div class="warn"><b>MEDIUM RISK</b> — score: {score:.4f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bad"><b>HIGH RISK</b> — score: {score:.4f}</div>', unsafe_allow_html=True)

        dfp = pd.DataFrame({"Model": list(parts.keys()), "Risk": list(parts.values())})
        fig = px.bar(dfp, x="Model", y="Risk", color="Model", title="Base model predictions",
                     color_discrete_sequence=["#6C63FF","#00d2d3","#f368e0","#10ac84","#ff9f43"])
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Current configuration")
    st.dataframe(pretty_table(base), use_container_width=True, hide_index=True)
