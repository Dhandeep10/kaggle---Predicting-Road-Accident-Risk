# Road Safety Prediction App

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF6B35?style=for-the-badge&logo=streamlit&logoColor=white)](https://kaggle---predicting-road-accident-risk-y538qmadtqtmiq97ggurl4.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)

An interactive **Road Safety Prediction App** built with **Streamlit** and **Machine Learning**. This project uses ensemble models (XGBoost, LightGBM, CatBoost, Extra Trees, Random Forest) to predict road accident risks based on real-world factors like road type, weather, lighting, speed limits, and historical accidents.

The app features three engaging tabs:
- **🎮 Game**: Test your road safety knowledge by choosing safer routes across 10 rounds.
- **🔍 Explore**: Interactive risk assessment with real-time model predictions.
- **🚦 Journey**: Multi-step journey simulation where risky choices end the trip early.

Perfect for **data science students**, **road safety researchers**, and anyone interested in **predictive modeling** for public safety.

## 🏗️ Project Structure

kaggle---Predicting-Road-Accident-Risk/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── model_xgb.pkl # XGBoost model
├── model_lgb.pkl # LightGBM model
├── model_cat.pkl # CatBoost model
├── model_et.pkl # Extra Trees model
├── model_rf.pkl # Random Forest model
├── meta.pkl # Meta-ensemble model
├── README.md


## 🚀 Features

- **Ensemble ML Models**: Combines 5 powerful algorithms for robust risk prediction (0-1 probability scale).
- **Interactive Game Mode**: 10-round quiz to compare road scenarios and learn safety patterns.
- **Real-time Risk Explorer**: Adjust parameters (weather, speed, curvature) and see model outputs.
- **Journey Simulation**: Navigate 3-8 steps; exceed risk threshold (0.6) and restart!
- **Performance Optimized**: Uses `@st.cache_resource` and `@lru_cache` for smooth experience.
- **Mobile-Friendly**: Responsive design works on all devices.

### Key Risk Factors Modeled
- **Road Conditions**: Type (highway/urban), lanes, curvature, speed limits
- **Environmental**: Weather (clear/rain/fog), lighting (day/twilight/dark), time of day
- **Contextual**: Holidays, school seasons, historical accidents, road signs presence
- **Derived Features**: 30+ engineered features like speed-curvature interaction, risk scores

## 🛠️ Tech Stack

### Core Technologies
- **Streamlit**: Interactive web app framework
- **Scikit-learn 1.2.2**: Base ML models and preprocessing
- **XGBoost 2.0.3**: Gradient boosting
- **LightGBM 4.6.0**: Fast gradient boosting
- **CatBoost 1.2.8**: Categorical feature handling
- **Pandas 1.5.3**: Data manipulation
- **NumPy 1.24.3**: Numerical computations
- **Joblib 1.3.2**: Model serialization

### Deployment
- **Streamlit Cloud**: Hosted on GitHub with automatic deployments
- **Python 3.11**: Compatible runtime (tested on Streamlit Cloud)

## 📊 Model Performance

The ensemble achieves high accuracy through stacking:

| Model          | Type          | Key Strength                  |
|----------------|---------------|-------------------------------|
| XGBoost       | Tree-based   | Handles interactions well    |
| LightGBM      | Tree-based   | Fast training & prediction   |
| CatBoost      | Tree-based   | Categorical features         |
| Extra Trees   | Tree-based   | Reduces overfitting          |
| Random Forest | Tree-based   | Baseline stability           |
| **Meta-Model**| Linear       | Combines predictions (0-1)   |

**Risk Threshold**: 0.6 (Medium-High risk) – Configurable in Journey tab.

## 🎯 Getting Started

### Prerequisites
- Python 3.8+
- Git installed

### Local Installation

1. **Clone the Repository**
git clone https://github.com/Dhandeep10/kaggle---Predicting-Road-Accident-Risk.git
cd kaggle---Predicting-Road-Accident-Risk

2. **Create Virtual Environment**
Using venv (recommended)
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Or using conda
conda create -n road-safety python=3.11
conda activate road-safety


3. **Install Dependencies**
pip install -r requirements.txt


4. **Run the App**
streamlit run app.py

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Model Files
- Download the `.pkl` model files from the repo or train your own using the Kaggle dataset.
- Place them in the root directory as listed in the structure.

## 🌐 Live Demo

Try the app live:  
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kaggle---predicting-road-accident-risk-y538qmadtqtmiq97ggurl4.streamlit.app/)

**Note**: If models fail to load, ensure `.pkl` files are present and compatible with your Python version.

## 📈 Usage Examples

### Game Tab (🎮)
1. Compare two road scenarios (A vs B).
2. Choose the safer route based on parameters.
3. Complete 10 rounds; aim for >70% accuracy to earn "Expert Driver" rating.

### Explore Tab (🔍)
1. Adjust sliders/dropdowns for road conditions.
2. Click "Calculate Risk" to see ensemble prediction.
3. View individual model contributions in a table.

### Journey Tab (🚦)
1. Set journey length (3-8 steps) and risk threshold (0.4-0.9).
2. At each step, pick from 3 routes (A/B/C).
3. If risk > threshold, journey ends – restart to improve!

## 🧪 Development & Troubleshooting

### Common Issues
- **Model Loading Error**: Ensure `.pkl` files match the scikit-learn version (1.2.2). Retrain if needed.
- **Pandas Version**: Use exactly `pandas==1.5.3` for compatibility with trained models.
- **Auto-Clicking Buttons**: Fixed with unique session keys; clear browser cache if persists.
- **Lag**: App uses caching; for local runs, ensure sufficient RAM (models are ~100MB total).

### Extending the Project
- **Add More Features**: Include traffic density, vehicle type, or GPS integration.
- **New Models**: Train with neural networks (e.g., via TensorFlow) and add to ensemble.
- **Dataset**: Based on Kaggle's road accident data; extend with real-time APIs (weather/traffic).
- **Animations**: Enhance Journey tab with Streamlit's `st.balloons()` or custom CSS for road visuals.


## 🤝 Contributing

Contributions welcome! 🚀

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

### Guidelines
- Follow PEP 8 for Python code.
- Add tests for new features.
- Update README if changes affect setup/usage.


## 👥 Acknowledgments

- **Built with**: Streamlit, Scikit-learn, XGBoost, LightGBM, CatBoost.
- **Dataset Inspiration**: Kaggle road accident datasets.
- **Deployment**: Streamlit Cloud for free hosting.
- **Creator**: Dhandeep Singh

## 📞 Contact

Dhandeep Singh  
- GitHub: [@Dhandeep10](https://github.com/Dhandeep10)  

**Project Status**: Actively maintained (Last update: October 2025).

---

*⭐ If this project helps you, star the repo to show support! ⭐*
