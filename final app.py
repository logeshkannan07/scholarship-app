# ---------------------------------------------------------------------
# FINAL STREAMLIT APP — SCHOLARSHIP FINDER + REACH PREDICTOR + DASHBOARD
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ------------------------- PAGE CONFIG -------------------------------
st.set_page_config(
    page_title="Smart Scholarship Finder & Reach Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        /* soft blue background */
        body, .stApp {
            background-color: #e8f2ff !important;
        }
        .stButton>button {
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: 600;
            transition: 0.2s;
        }
        .stButton>button:hover {
            transform: scale(1.03);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- LOAD DATASETS -----------------------------
sch_df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
reach_df = pd.read_csv("TN_Scholarship_Reach_REALISTIC.csv")

# numeric income column
if 'income_limit_numeric' not in sch_df.columns:
    sch_df['income_limit_numeric'] = (
        sch_df['income_limit']
        .replace('[^0-9]', '', regex=True)
        .replace('', np.nan)
        .astype(float)
    )

# load ML models
with open("Linear_Regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("Random_Forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("Gradient_Boosting_model.pkl", "rb") as f:
    gb_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------------- APP TITLE --------------------------------
st.title("🎓 Smart Scholarship Finder & Reach Predictor")
st.markdown("An AI-powered platform to discover scholarships, check your eligibility, and visualize insights.")

# ------------------------- TABS -------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Eligibility Finder", "🤖 Reach Predictor", "📈 Data Dashboard"])

# =====================================================================
# TAB 1 — ELIGIBILITY FINDER (simplified filters)
# =====================================================================
with tab1:
    st.header("🎯 Scholarship Eligibility Finder")
    st.markdown("Enter your details to find scholarships you qualify for. Results appear as clickable cards.")

    genders = ["All", "Male", "Female"]
    categories = ["All", "SC", "ST", "OBC", "General"]
    edulevels = ["All", "School", "Undergraduate", "Postgraduate", "PhD"]

    col1, col2, col3 = st.columns(3)
    with col1:
        input_gender = st.selectbox("Gender", genders, index=0)
        input_category = st.selectbox("Category", categories, index=0)
    with col2:
        input_income = st.number_input("Annual Family Income (₹)", min_value=0, value=150000, step=5000)
        input_edu = st.selectbox("Education Level", edulevels, index=0)
    with col3:
        search_term = st.text_input("Search scholarship name or keyword", "")

    def filter_eligible(df):
        df2 = df.copy()
        if 'income_limit_numeric' in df2.columns:
            df2 = df2[(df2['income_limit_numeric'].isna()) | (input_income <= df2['income_limit_numeric'])]
        if input_gender != "All":
            df2 = df2[df2['gender'].str.contains(input_gender, case=False, na=False)]
        if input_category != "All":
            df2 = df2[df2['category'].str.contains(input_category, case=False, na=False)]
        if input_edu != "All":
            df2 = df2[df2['education_level'].str.contains(input_edu, case=False, na=False)]
        if search_term:
            df2 = df2[df2['scholarship_name'].str.contains(search_term, case=False, na=False)]
        return df2

    if st.button("🔎 Find Eligible Scholarships"):
        eligible_df = filter_eligible(sch_df)
        st.session_state['eligible_df'] = eligible_df

        if eligible_df.empty:
            st.warning("No matching scholarships found.")
        else:
            st.success(f"Found {len(eligible_df)} eligible scholarships.")
            for _, row in eligible_df.iterrows():
                with st.expander(f"🎓 {row['scholarship_name']}"):
                    st.write(f"**Provider:** {row.get('provider','N/A')}")
                    st.write(f"**Category:** {row.get('category','N/A')}")
                    st.write(f"**Gender:** {row.get('gender','N/A')}")
                    st.write(f"**Education Level:** {row.get('education_level','N/A')}")
                    st.write(f"**Income Limit:** {row.get('income_limit','N/A')}")
                    st.write(f"**Level:** {row.get('level','N/A')}")
                    st.write(f"**Official Link:** {row.get('official_website','N/A')}")

# =====================================================================
# TAB 2 — REACH PREDICTOR (unchanged)
# =====================================================================
with tab2:
    st.header("🤖 Scholarship Reach Predictor (Model Comparison)")

    col1, col2 = st.columns(2)
    with col1:
        applicants = st.number_input("Number of Applicants", min_value=0, value=1000)
        budget = st.number_input("Scholarship Budget (₹)", min_value=0, value=5000000)
    with col2:
        awareness = st.slider("Awareness Index (0–100)", 0, 100, 60)
        outreach = st.slider("Outreach Score (0–100)", 0, 100, 70)

    X_input = pd.DataFrame([[applicants, budget, awareness, outreach]],
                           columns=["Applicants", "Budget", "Awareness", "Outreach"])
    X_scaled = scaler.transform(X_input)

    lr_pred = lr_model.predict(X_scaled)[0]
    rf_pred = rf_model.predict(X_scaled)[0]
    gb_pred = gb_model.predict(X_scaled)[0]

    st.subheader("Predicted Reach (%) by Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Linear Regression", f"{lr_pred:.2f}%")
    c2.metric("Random Forest", f"{rf_pred:.2f}%")
    c3.metric("Gradient Boosting", f"{gb_pred:.2f}%")

    st.markdown("---")
    st.write("Average predicted reach:", f"**{(lr_pred + rf_pred + gb_pred)/3:.2f}%**")

# =====================================================================
# TAB 3 — DATA DASHBOARD (full dataset + advanced filters)
# =====================================================================
with tab3:
    st.header("📈 Dataset Visualization Dashboard")
    st.markdown("Explore the **entire dataset** of scholarships using filters below.")

    df_vis = sch_df.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        level_filter = st.multiselect("Filter by Level",
                                      sorted(df_vis['level'].dropna().unique()),
                                      default=sorted(df_vis['level'].dropna().unique()))
        cat_filter = st.multiselect("Filter by Category",
                                    ["All", "SC", "ST", "OBC", "General"], default=["All"])
    with col2:
        gender_filter = st.multiselect("Filter by Gender", ["All", "Male", "Female"], default=["All"])
        edu_filter = st.multiselect("Filter by Education Level",
                                    ["All", "School", "Undergraduate", "Postgraduate", "PhD"], default=["All"])
    with col3:
        max_income = int(df_vis['income_limit_numeric'].max() or 1000000)
        income_range = st.slider("Income Limit (₹)", 0, max_income, (0, max_income))
        show_top = st.slider("Top N Categories (for bar chart)", 3, 30, 10)

    # Apply filters
    if "All" not in cat_filter:
        df_vis = df_vis[df_vis['category'].isin(cat_filter)]
    if "All" not in gender_filter:
        df_vis = df_vis[df_vis['gender'].isin(gender_filter)]
    if "All" not in edu_filter:
        df_vis = df_vis[df_vis['education_level'].isin(edu_filter)]
    df_vis = df_vis[df_vis['level'].isin(level_filter)]
    df_vis = df_vis[(df_vis['income_limit_numeric'] >= income_range[0]) &
                    (df_vis['income_limit_numeric'] <= income_range[1])]

    st.subheader("Summary Metrics")
    t1, t2, t3 = st.columns(3)
    t1.metric("Total Scholarships", len(df_vis))
    t2.metric("Tamil Nadu (approx)",
              len(df_vis[df_vis['level'].str.contains("state|tn|tamil", case=False, na=False)]))
    t3.metric("Central (approx)",
              len(df_vis[df_vis['level'].str.contains("central|national", case=False, na=False)]))

    st.markdown("---")

    # Charts
    if not df_vis.empty:
        st.subheader("Bar Chart — Top Categories")
        vc = df_vis['category'].value_counts().nlargest(show_top).reset_index()
        vc.columns = ['Category', 'Count']
        fig = px.bar(vc, x='Category', y='Count', title=f"Top {show_top} Categories")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pie Chart — Level Distribution")
        lvl = df_vis['level'].value_counts().reset_index()
        lvl.columns = ['Level', 'Count']
        fig2 = px.pie(lvl, values='Count', names='Level', title="Level Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data available for visualization.")

    st.markdown("---")
    st.subheader("Filtered Data View")
    st.dataframe(df_vis.reset_index(drop=True), use_container_width=True)

# ---------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------


# ---------------- Footer ----------------
st.markdown("---")
st.markdown("**Developed by:** Logesh Kannan  ·  **Guide:** Dr. Rajkumar  · Anna University Regional Campus, Madurai")








