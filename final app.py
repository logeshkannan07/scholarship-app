import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os

# ================================
# APP CONFIGURATION
# ================================
st.set_page_config(page_title="Scholarship Finder Dashboard", layout="wide")

st.title("🎓 Anna University Scholarship Finder & Reach Dashboard")
st.markdown("---")

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
        df.fillna("N/A", inplace=True)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found! Please upload 'Curated_Scholarships_India_TN_200.csv'.")
        st.stop()

df = load_data()

# ================================
# LOAD MULTIPLE MODELS & SCALER
# ================================
@st.cache_data
def load_all_models():
    models = {}
    model_files = {
        "Linear Regression": "Linear_Regression_model.pkl",
        "Random Forest": "Random_Forest_model.pkl",
        "Gradient Boosting": "Gradient_Boosting_model.pkl"
    }
    scaler_path = "scaler.pkl"

    # Load scaler
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        st.warning("⚠️ Scaler file 'scaler.pkl' not found.")

    # Load all models
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            st.warning(f"⚠️ Model file '{path}' not found.")

    if models:
        st.success(f"✅ Loaded {len(models)} model(s) successfully.")
    else:
        st.error("❌ No models found. Please upload your .pkl model files.")
        st.stop()

    return models, scaler

models, scaler = load_all_models()

# ================================
# CREATE TABS
# ================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Scholarship Eligibility Finder",
    "📊 Reach Prediction",
    "📈 Dataset Dashboard"
])

# ================================
# TAB 1: ELIGIBILITY CHECKER
# ================================
with tab1:
    st.header("🎓 Find Your Eligible Scholarships")

    gender = st.selectbox("Select Gender", ["All", "Male", "Female"])
    category = st.selectbox("Select Category", df["Category"].unique())
    income_limit = st.number_input("Enter Annual Income", min_value=0)
    education_level = st.selectbox("Select Education Level", df["Education Level"].unique())

    if st.button("🔍 Check Eligibility"):
        filtered_df = df.copy()
        filtered_df = filtered_df[
            (filtered_df["Category"] == category) &
            ((filtered_df["Gender"] == gender) | (filtered_df["Gender"] == "All")) &
            (filtered_df["Education Level"] == education_level) &
            (filtered_df["Income Limit"].astype(str).apply(lambda x: x.replace(",", "").replace("₹", "").replace("-", "").strip().isdigit()))  # Clean income
        ]

        # Split TN & Central Scholarships
        tn_sch = filtered_df[filtered_df["Level"].str.contains("State", case=False, na=False)]
        central_sch = filtered_df[filtered_df["Level"].str.contains("Central", case=False, na=False)]

        st.subheader("🏛️ Tamil Nadu Scholarships")
        if not tn_sch.empty:
            for _, row in tn_sch.iterrows():
                st.markdown(f"""
                <div style="background-color:#f0f8ff;padding:10px;border-radius:10px;margin-bottom:8px;">
                <b>{row['Scholarship Name']}</b><br>
                💰 Amount: {row['Amount']}<br>
                🎓 Education Level: {row['Education Level']}<br>
                🌐 <a href="{row['Website']}" target="_blank">Apply Here</a><br>
                📝 {row['Description']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No Tamil Nadu scholarships found for this criteria.")

        st.subheader("🇮🇳 Central Government Scholarships")
        if not central_sch.empty:
            for _, row in central_sch.iterrows():
                st.markdown(f"""
                <div style="background-color:#fff3cd;padding:10px;border-radius:10px;margin-bottom:8px;">
                <b>{row['Scholarship Name']}</b><br>
                💰 Amount: {row['Amount']}<br>
                🎓 Education Level: {row['Education Level']}<br>
                🌐 <a href="{row['Website']}" target="_blank">Apply Here</a><br>
                📝 {row['Description']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No Central scholarships found for this criteria.")

# ================================
# TAB 2: SCHOLARSHIP REACH PREDICTION
# ================================
with tab2:
    st.header("📊 Predict Scholarship Reach")

    model_choice = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_choice]

    st.info(f"Using model: **{model_choice}**")

    district = st.text_input("Enter District Name")
    income = st.number_input("Enter Average Student Income (₹)", min_value=0)
    education_score = st.slider("Education Index (0 to 1)", 0.0, 1.0, 0.5)

    if st.button("🚀 Predict Reach Rate"):
        if scaler:
            features = scaler.transform([[income, education_score]])
        else:
            features = [[income, education_score]]

        prediction = model.predict(features)
        reach_rate = round(prediction[0], 2)

        st.success(f"📈 Predicted Scholarship Reach Rate for {district}: **{reach_rate}%**")

# ================================
# TAB 3: DATASET DASHBOARD
# ================================
with tab3:
    st.header("📊 Visualize the Scholarship Dataset")

    level = st.multiselect("Filter by Level", df["Level"].unique(), default=df["Level"].unique())
    category_filter = st.multiselect("Filter by Category", df["Category"].unique(), default=df["Category"].unique())

    df_dash = df[(df["Level"].isin(level)) & (df["Category"].isin(category_filter))]

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(df_dash, x="Category", color="Level", title="Scholarships by Category & Level", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        level_count = df_dash["Level"].value_counts().reset_index()
        level_count.columns = ["Level", "Count"]
        fig2 = px.pie(level_count, values="Count", names="Level", title="Distribution of Scholarships by Level")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📄 Preview of Dataset")
    st.dataframe(df_dash, use_container_width=True)

