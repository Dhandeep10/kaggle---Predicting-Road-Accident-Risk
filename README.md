# Road Safety Predictor

Interactive web application for the Kaggle October Playground Series S5E10 challenge. Built with your ensemble model to predict road accident risk and educate users through engaging gameplay.

## Features

### 🎮 Game - Pick the Safer Road
- 10-round challenge comparing two randomized roads at a time
- Test your intuition against the trained ensemble (XGBoost, LightGBM, CatBoost, ExtraTrees, RandomForest + meta-learner)
- Score your choices and receive immediate feedback after each round
- Final summary shows accuracy and common mistake patterns

### 🔎 Explore a Single Road
- Configure road parameters (type, lanes, curvature, speed, lighting, weather, etc.)
- Get ensemble predictions with base model contributions
- See recommended interventions and their expected risk reduction
- Real-time updates as you adjust parameters

### 🚦 Journey (Start to Finish)
- Sequential decision-making game with 3-8 checkpoints
- Choose from 2-3 options at each checkpoint without knowing risks
- Survive all checkpoints to reach destination, or restart on threshold breach
- End summary reveals actual risks, optimal path, and improvement suggestions
- Adjustable difficulty (threshold, steps, options per checkpoint)

## Installation

1. Clone this repository:
git clone https://github.com/Dhandeep10/kaggle---Predicting-Road-Accident-Risk.git
cd kaggle---Predicting-Road-Accident-Risk


2. Install dependencies:
pip install -r requirements.txt


3. Ensure model files exist:
- Save your trained ensemble models as: `model_xgb.pkl`, `model_lgb.pkl`, `model_cat.pkl`, `model_et.pkl`, `model_rf.pkl`, `meta.pkl`
- Place them in the root directory alongside `app.py`

4. Run locally:
streamlit run app.py


## Usage

The app launches three interactive tabs with different approaches to explore the road safety model:

### Game Mode
- Compare two randomized road configurations
- Choose the road you think is safer
- Receive immediate feedback and advance
- Complete 10 rounds to see final accuracy

### Explore Mode
- Adjust all 12 road parameters using sliders and dropdowns
- Click "Predict risk" to see ensemble output and individual model predictions
- View recommended interventions with estimated risk improvements

### Journey Mode
- Navigate 3-8 checkpoints representing a complete trip
- Select from multiple options at each step without seeing risk values
- Survive all checkpoints without exceeding the risk threshold
- Receive detailed analysis comparing your path to the optimal path

## Model Architecture

The app uses a stacking ensemble built in your Kaggle notebook:

**Base Models:**
- XGBoost - trained on original + engineered features
- LightGBM - optimized gradient boosting
- CatBoost - handles categorical features internally
- ExtraTrees - ensemble tree diversity
- RandomForest - classical ensemble component

**Meta-learner:** LightGBM model trained on base model predictions, learning to combine ensemble outputs for final risk score.

**Feature Engineering:** Matches exactly with your notebook:
- 12 base features (road_type, num_lanes, curvature, speed_limit, lighting, weather, road_signs_present, public_road, time_of_day, holiday, school_season, num_reported_accidents)
- 36+ engineered interactions (speed×curvature, lane×speed, weather×lighting, etc.)
- Risk output normalized between 0.0 (safest) and 1.0 (highest risk)

## Technical Details

### Dependencies
- Python 3.10+ 
- Streamlit 1.28.0 for the web interface
- Ensemble models saved with joblib 1.3.2+ for compatibility
- No external APIs or databases - runs offline with local model files

### Performance Optimizations
- `@st.cache_resource` for model loading (single instance)
- `@lru_cache` for repeated predictions on identical feature combinations
- Minimal UI updates - only re-renders when user actions occur
- Efficient session state management to preserve game progress
- Lightweight CSS animations (no heavy JavaScript)

### Session State Architecture
- Game: g_round, g_score, g_pair, g_hist
- Explore: stateless (parameters reset on interaction)
- Journey: j_step, j_thr, j_max, j_alive, j_hist, j_opts, j_seed

### File Structure
├── app.py # Main Streamlit application
├── model_xgb.pkl # XGBoost base model
├── model_lgb.pkl # LightGBM base model
├── model_cat.pkl # CatBoost base model
├── model_et.pkl # ExtraTrees base model
├── model_rf.pkl # RandomForest base model
├── meta.pkl # Meta-learner (LightGBM)
├── requirements.txt # Python dependencies
└── README.md 

