# ======================================================
# app.py
# ------------------------------------------------------
# Streamlit App for Coffee Rating Prediction
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="☕ Coffee Rating Predictor",
    page_icon="☕",
    layout="wide"
)

# === Animated Coffee Emoji Decorations ===
st.markdown("""
<style>
/* === Background Gradient === */
body {
    background: linear-gradient(135deg, #fff8e7, #f7e7ce);
    animation: bgmove 20s ease infinite;
}
@keyframes bgmove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* === Coffee Emoji Falling Animation === */
@keyframes fall {
    0% {
        transform: translateY(-10vh) rotate(0deg);
        opacity: 1;
    }
    100% {
        transform: translateY(110vh) rotate(360deg);
        opacity: 0;
    }
}

.coffee {
    position: fixed;
    top: -10vh;
    font-size: 28px;
    animation: fall linear infinite;
    z-index: 1;
}

/* Randomize positions, delays, speeds */
.coffee:nth-child(1)  { left: 5%;  animation-duration: 8s; animation-delay: 0s; }
.coffee:nth-child(2)  { left: 15%; animation-duration: 10s; animation-delay: 2s; }
.coffee:nth-child(3)  { left: 25%; animation-duration: 9s; animation-delay: 4s; }
.coffee:nth-child(4)  { left: 35%; animation-duration: 11s; animation-delay: 1s; }
.coffee:nth-child(5)  { left: 50%; animation-duration: 8s; animation-delay: 3s; }
.coffee:nth-child(6)  { left: 65%; animation-duration: 10s; animation-delay: 5s; }
.coffee:nth-child(7)  { left: 75%; animation-duration: 9s; animation-delay: 1s; }
.coffee:nth-child(8)  { left: 85%; animation-duration: 12s; animation-delay: 2s; }
.coffee:nth-child(9)  { left: 92%; animation-duration: 13s; animation-delay: 0s; }

/* Text Styling */
h1, h2, h3 {
    font-family: "Poppins", sans-serif;
    color: #4b2e05;
}
</style>

<!-- Falling Coffee Emojis -->
<div class="coffee">☕</div>
<div class="coffee">☕</div>
<div class="coffee">☕</div>
<div class="coffee">☕</div>
<div class="coffee">☕</div>
<div class="coffee">☕</div>
<div class="coffee">☕</div>
<div class="coffee">☕</div>
<div class="coffee">☕</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Load data and models
# ------------------------------------------------------
URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(URL)
    return data

@st.cache_resource
def load_models():
    with open("model_1.pickle", "rb") as f1:
        model_1 = pickle.load(f1)
    with open("model_2.pickle", "rb") as f2:
        model_2 = pickle.load(f2)
    return model_1, model_2

data = load_data()
model_1, model_2 = load_models()

# ------------------------------------------------------
# Helper function
# ------------------------------------------------------
def roast_category(roast):
    """Convert roast type to numeric label (must match training)."""
    roast_map = {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }
    return roast_map.get(roast, None)

# ------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------
st.sidebar.title("☕ Coffee App Menu")
page = st.sidebar.radio("Navigate to:", ["🏠 Overview", "📊 Visualizations", "🤖 Predictions"])

# ------------------------------------------------------
# Overview Page
# ------------------------------------------------------
if page == "🏠 Overview":
    st.title("☕ Coffee Rating Analysis App")
    st.markdown("""
    Welcome!  
    This Streamlit app demonstrates coffee rating prediction using two ML models:
    - **Model 1:** Linear Regression (`100g_USD` → `rating`)
    - **Model 2:** Decision Tree Regressor (`100g_USD` + `roast` → `rating`)
    """)

    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head(10))

    st.success("✅ Models loaded successfully and ready to use!")

# ------------------------------------------------------
# Visualization Page
# ------------------------------------------------------
elif page == "📊 Visualizations":
    st.title("📊 Coffee Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Ratings")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data['rating'], bins=20, kde=True, color='saddlebrown', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Price vs Rating Scatter")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=data, x='100g_USD', y='rating', hue='roast', palette='viridis', ax=ax)
        st.pyplot(fig)

    st.subheader("Average Rating by Roast Type")
    avg_ratings = data.groupby('roast')['rating'].mean().sort_values(ascending=False)
    st.bar_chart(avg_ratings)

# ------------------------------------------------------
# Prediction Page
# ------------------------------------------------------
elif page == "🤖 Predictions":
    st.title("🤖 Predict Coffee Rating")

    st.write("Adjust inputs below to predict a coffee’s rating.")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("💰 Price per 100g (USD)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)

    with col2:
        roast = st.selectbox("🔥 Roast Type", ["Light", "Medium-Light", "Medium", "Medium-Dark", "Dark"])

    roast_cat = roast_category(roast)

    model_choice = st.radio(
        "Select Model for Prediction",
        ["Model 1 - Linear Regression", "Model 2 - Decision Tree Regressor"]
    )

    if st.button("🔮 Predict Rating"):
        if model_choice == "Model 1 - Linear Regression":
            X_pred = np.array([[price]])
            pred = model_1.predict(X_pred)[0]
        else:
            X_pred = np.array([[price, roast_cat]])
            pred = model_2.predict(X_pred)[0]

        st.success(f"☕ Predicted Coffee Rating: **{pred:.2f} / 100**")

        # Show on scatter plot
        st.subheader("📈 Comparison with Dataset")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=data, x='100g_USD', y='rating', alpha=0.5, color='gray', label='Existing Data')
        ax.scatter(price, pred, color='red', s=100, label='Your Prediction')
        ax.set_xlabel("Price per 100g (USD)")
        ax.set_ylabel("Rating")
        ax.legend()
        st.pyplot(fig)

# ------------------------------------------------------
# Footer
# ------------------------------------------------------
st.markdown("---")
