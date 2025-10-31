# =========================================================
# 🎓 Anna University Scholarship App (Enhanced & Error-Free)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 🌐 App Config
# ---------------------------------------------------------
st.set_page_config(page_title="Scholarship Dashboard", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .card {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    a {
        color: #0d6efd !important;
        text-decoration: none;
        font-weight: 600;
    }
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

st.image("anna-university-logo.png", width=140)
st.title("🎓 Anna University Scholarship App")
st.markdown("#### Scholarship Eligibility Finder | Reach Prediction | Data Dashboard")
st.markdown("---")

# ---------------------------------------------------------
# 🧠 Safe Model Loading
# ---------------------------------------------------------
@st.cache_data
def load_all_models():
    models = {}
    try:
        models["Linear Regression"] = joblib.load("Linear_Regression_model.pkl")
    except:
        st.warning("⚠️ Linear Regression model not found.")
    try:
        models["Random Forest"] = joblib.load("Random_Forest_model.pkl")
    except:
        st.warning("⚠️ Random Forest model not found.")
    try:
        models["Gradient Boosting"] = joblib.load("Gradient_Boosting_model.pkl")
    except:
        st.warning("⚠️ Gradient Boosting model not found.")
    # Safe Scaler Loading
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Scaler not loaded: {e}. Using unscaled inputs.")
        scaler = None
    return models, scaler

models, scaler = load_all_models()

# ---------------------------------------------------------
# 🏷️ Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏆 Scholarship Eligibility Finder",
    "📊 Scholarship Reach Predictor",
    "📈 Data Dashboard"
])

# =========================================================
# TAB 1 — Scholarship Eligibility Finder
# =========================================================
with tab1:
    st.subheader("Find Your Eligible Scholarships")

    @st.cache_data
    def load_scholarship_data():
        df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
        df.columns = df.columns.str.strip()
        return df

    df_sch = load_scholarship_data()

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        category = st.selectbox("Community / Category", ["SC", "ST", "OBC", "General", "Minority"])
    with col2:
        income = st.number_input("Annual Family Income (₹)", min_value=0, value=200000)
        edu_level = st.selectbox("Education Level", ["School", "UG", "PG", "PhD"])

    if st.button("🔍 Find Eligible Scholarships"):
        def check_eligibility(row):
            if row["Gender"] != "All" and row["Gender"].lower() != gender.lower():
                return False
            if row["Category"] != "All" and row["Category"].upper() != category.upper():
                return False
            try:
                limit = float(str(row["Income_Limit"]).replace(",", "").replace("₹", "").strip())
                if income > limit:
                    return False
            except:
                pass
            if row["Eligible_Classes"] != "All" and edu_level.lower() not in row["Eligible_Classes"].lower():
                return False
            return True

        result_df = df_sch[df_sch.apply(check_eligibility, axis=1)]

        if not result_df.empty:
            st.success(f"✅ Found {len(result_df)} eligible scholarships for you!")

            # Separate by Level (State vs Central)
            tn_df = result_df[result_df["Level"].str.contains("State", case=False, na=False)]
            central_df = result_df[result_df["Level"].str.contains("Central", case=False, na=False)]

            # Tamil Nadu Scholarships Section
            if not tn_df.empty:
                st.markdown("### 🎯 Tamil Nadu Level Scholarships")
                for _, row in tn_df.iterrows():
                    st.markdown(f"""
                    <div class="card">
                        <b>{row['Scholarship_Name']}</b><br>
                        Provider: {row.get('Provider', 'N/A')}<br>
                        Amount: ₹{row.get('Amount', 'N/A')}<br>
                        <a href="{row.get('Application_Link', '#')}" target="_blank">🔗 Apply Here</a>
                    </div>
                    """, unsafe_allow_html=True)

            # Central Scholarships Section
            if not central_df.empty:
                st.markdown("### 🏛️ Central Level Scholarships")
                for _, row in central_df.iterrows():
                    st.markdown(f"""
                    <div class="card" style="background-color:#fff3cd;">
                        <b>{row['Scholarship_Name']}</b><br>
                        Provider: {row.get('Provider', 'N/A')}<br>
                        Amount: ₹{row.get('Amount', 'N/A')}<br>
                        <a href="{row.get('Application_Link', '#')}" target="_blank">🔗 Apply Here</a>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.warning("😔 No matching scholarships found.")

# =========================================================
# TAB 2 — Scholarship Reach Predictor
# =========================================================
with tab2:
    st.subheader("Predict Scholarship Reach (District-wise)")
    try:
        df_reach = pd.read_csv("TN_Scholarship_Reach_REALISTIC.csv")
    except:
        st.error("Dataset TN_Scholarship_Reach_REALISTIC.csv not found.")
        st.stop()

    district = st.selectbox("Select District", df_reach["district"].unique())
    model_choice = st.selectbox("Choose Model", list(models.keys()))

    district_data = df_reach[df_reach["district"] == district].iloc[0]
    X_cols = ["avg_family_income", "literacy_rate", "female_ratio", "rural_population_percent",
              "num_students", "schools_with_computer_lab_percent", "schools_with_internet_percent",
              "school_infrastructure_index"]

    inputs = [st.number_input(f"{col}", value=float(district_data[col])) for col in X_cols]
    features = np.array([inputs])

    if scaler is not None:
        try:
            features = scaler.transform(features)
        except:
            st.warning("⚠️ Error scaling inputs; using unscaled values.")

    if st.button("🚀 Predict Reach"):
        model = models.get(model_choice)
        if model:
            pred = model.predict(features)[0]
            st.success(f"🎯 Predicted Scholarship Reach in {district}: **{pred:.2f}%**")
        else:
            st.warning("Selected model not available.")

# =========================================================
# TAB 3 — Data Dashboard
# =========================================================
with tab3:
    st.subheader("📈 Scholarship Dataset Overview")

    df_dash = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
    st.dataframe(df_dash, use_container_width=True)

    # Visualization
    col1, col2 = st.columns(2)
    with col1:
        col_select = st.selectbox("Select Column for Bar Chart", ["Category", "Gender", "Level"])
        fig = px.bar(df_dash[col_select].value_counts(), title=f"{col_select} Distribution", color=df_dash[col_select].value_counts().index)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        pie_col = st.selectbox("Select Column for Pie Chart", ["Category", "Level", "Gender"])
        fig2 = px.pie(df_dash, names=pie_col, title=f"{pie_col} Breakdown")
        st.plotly_chart(fig2, use_container_width=True)

    # Histogram
    st.subheader("💰 Scholarship Amount Distribution")
    try:
        df_dash["Amount_num"] = df_dash["Amount"].astype(str).str.replace("[₹,–]", "", regex=True).replace("", "0").astype(float)
        fig3 = px.histogram(df_dash, x="Amount_num", nbins=30, title="Distribution of Scholarship Amounts")
        st.plotly_chart(fig3, use_container_width=True)
    except:
        st.warning("Could not display histogram for amounts.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("**Developed by:** Logesh Kannan S | **Guided by:** Anna University Regional Campus, Madurai")


