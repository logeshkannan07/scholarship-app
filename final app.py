# ================== FINAL FULL STREAMLIT APP ==================
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Scholarship Eligibility & Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #e8f1fa;
}
h1, h2, h3, h4 {
    color: #003366;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Add helper columns
    if 'income_limit' in df.columns:
        df['income_limit_numeric'] = (
            df['income_limit']
            .astype(str)
            .str.replace(r'[^\d]', '', regex=True)
            .replace('', np.nan)
            .astype(float)
        )

    df['gender_norm'] = df['gender'].fillna('All').str.strip().str.title()
    df['provider_type'] = df['provider'].fillna('').apply(lambda x: "Tamil Nadu" if "tamil" in x.lower() else ("Central" if "central" in x.lower() else "Other"))

    # Category & Education sets
    df['category_tags'] = df['category'].astype(str).str.lower().str.split(r'[,/ ]+')
    df['edu_tags'] = df['education_level'].astype(str).str.lower().str.split(r'[,/ ]+')
    return df

sch_df = load_dataset()

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    models = {}
    models["Linear Regression"] = pickle.load(open("Linear_Regression_model.pkl", "rb"))
    models["Gradient Boosting"] = pickle.load(open("Gradient_Boosting_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return models, scaler

models, scaler = load_models()

# -------------------- APP TABS --------------------
tab1, tab2, tab3 = st.tabs(["🎯 Eligibility Finder", "📈 Reach Predictor", "📊 Dashboard"])

# -------------------- TAB 1: ELIGIBILITY FINDER --------------------
with tab1:
    st.header("🎯 Scholarship Eligibility Finder")
    st.markdown("Enter your details in the sidebar to find eligible scholarships.")

    # Sidebar inputs
    st.sidebar.header("🧩 Enter Your Details")
    gender = st.sidebar.selectbox("Gender", ["All", "Male", "Female", "Transgender"])
    category = st.sidebar.selectbox("Category", ["All", "BC/OBC", "MBC", "SC", "ST", "General"])
    education = st.sidebar.selectbox("Education Level", ["All", "School", "UG", "PG", "PhD"])
    income = st.sidebar.number_input("Annual Family Income (₹)", min_value=0, value=150000, step=5000)
    search_term = st.sidebar.text_input("Search Scholarship or Keyword", "")

    def filter_eligible(df):
        df2 = df.copy()

        # Income
        if 'income_limit_numeric' in df2.columns:
            df2 = df2[(df2['income_limit_numeric'].isna()) | (income <= df2['income_limit_numeric'])]

        # Gender
        if gender != "All":
            if gender == "Male":
                df2 = df2[~df2['gender_norm'].eq('Female')]
            elif gender == "Female":
                df2 = df2[df2['gender_norm'].isin(['Female', 'All'])]
            elif gender == "Transgender":
                df2 = df2[df2['gender_norm'].isin(['Transgender', 'All'])]

        # Category
        if category != "All":
            sel = category.lower()
            def cat_match(tags):
                if not isinstance(tags, (set, list)): return False
                t = set([str(x).lower() for x in tags])
                if sel in ['bc/obc', 'bc', 'obc']:
                    return 'bc' in t or 'obc' in t
                if sel == 'sc':
                    return 'sc' in t or 'minority' in t
                if sel == 'st':
                    return 'st' in t or 'minority' in t
                if sel == 'mbc':
                    return 'mbc' in t
                if sel == 'general':
                    return 'general' in t
                return sel in t
            df2 = df2[df2['category_tags'].apply(cat_match)]

        # Education
        if education != "All":
            sel = education.lower()
            def edu_match(tags):
                if not isinstance(tags, (set, list)): return False
                t = set([str(x).lower() for x in tags])
                if sel == 'ug':
                    return ('ug' in t) or ('ug/pg' in t) or ('all' in t)
                if sel == 'pg':
                    return ('pg' in t) or ('ug/pg' in t) or ('all' in t)
                if sel == 'school':
                    return ('school' in t) or ('all' in t)
                if sel == 'phd':
                    return ('phd' in t) or ('all' in t)
                return False
            df2 = df2[df2['edu_tags'].apply(edu_match)]

        # Search term
        if search_term:
            df2 = df2[
                df2['scholarship_name'].str.contains(search_term, case=False, na=False) |
                df2['description'].str.contains(search_term, case=False, na=False)
            ]
        return df2

    if st.sidebar.button("🔎 Find Eligible Scholarships"):
        eligible_df = filter_eligible(sch_df)
        st.session_state['eligible_df'] = eligible_df

        if eligible_df.empty:
            st.warning("No scholarships matched your criteria.")
        else:
            st.success(f"Found {len(eligible_df)} scholarships.")
            tn_df = eligible_df[eligible_df['provider_type'].str.contains("Tamil Nadu", case=False, na=False)]
            central_df = eligible_df[eligible_df['provider_type'].str.contains("Central", case=False, na=False)]

            c1, c2, c3 = st.columns(3)
            c1.metric("🏛️ Tamil Nadu", len(tn_df))
            c2.metric("🇮🇳 Central", len(central_df))
            c3.metric("📄 Total", len(eligible_df))

            st.markdown("---")
            view_mode = st.radio("View as", ["Cards", "Table"], horizontal=True)

            if view_mode == "Table":
                show_cols = ['scholarship_name', 'education_level', 'category', 'gender', 'income_limit', 'amount', 'website']
                st.dataframe(eligible_df[show_cols].reset_index(drop=True), use_container_width=True)
            else:
                st.markdown("### 🟢 Tamil Nadu Scholarships")
                for _, r in tn_df.iterrows():
                    web = r.get('website', '')
                    if web and not web.lower().startswith("http"):
                        web = "https://" + web
                    st.markdown(f"""
                        <div style='background:#f0faff;padding:12px;border-radius:10px;margin-bottom:10px;'>
                            <h4 style='color:#0a58ca'>{r.get('scholarship_name','')}</h4>
                            <p><b>Category:</b> {r.get('category','')} | <b>Gender:</b> {r.get('gender','')} | <b>Edu:</b> {r.get('education_level','')}</p>
                            <p><b>Income Limit:</b> {r.get('income_limit','')} | <b>Amount:</b> {r.get('amount','')}</p>
                            <a href="{web}" target="_blank" style='background:#198754;color:white;padding:6px 10px;border-radius:8px;text-decoration:none;'>Apply</a>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 🟣 Central Scholarships")
                for _, r in central_df.iterrows():
                    web = r.get('website', '')
                    if web and not web.lower().startswith("http"):
                        web = "https://" + web
                    st.markdown(f"""
                        <div style='background:#fdf2ff;padding:12px;border-radius:10px;margin-bottom:10px;'>
                            <h4 style='color:#7b1fa2'>{r.get('scholarship_name','')}</h4>
                            <p><b>Category:</b> {r.get('category','')} | <b>Gender:</b> {r.get('gender','')} | <b>Edu:</b> {r.get('education_level','')}</p>
                            <p><b>Income Limit:</b> {r.get('income_limit','')} | <b>Amount:</b> {r.get('amount','')}</p>
                            <a href="{web}" target="_blank" style='background:#0d6efd;color:white;padding:6px 10px;border-radius:8px;text-decoration:none;'>Apply</a>
                        </div>
                    """, unsafe_allow_html=True)

# -------------------- TAB 2: REACH PREDICTOR --------------------
with tab2:
    st.header("📈 Scholarship Reach Predictor")
    st.write("Predict the reach or popularity score of scholarships using trained ML models.")

    model_name = st.selectbox("Select Model", ["Linear Regression", "Gradient Boosting"])
    model = models[model_name]

    reach_inputs = st.columns(2)
    with reach_inputs[0]:
        num_sch = st.number_input("Number of Applicants", min_value=0, value=500)
        amt = st.number_input("Scholarship Amount (₹)", min_value=0, value=10000)
    with reach_inputs[1]:
        income_limit = st.number_input("Income Limit (₹)", min_value=0, value=200000)
        level = st.selectbox("Level", ["School", "UG", "PG", "PhD"])

    X = np.array([[num_sch, amt, income_limit, ["School","UG","PG","PhD"].index(level)]]).astype(float)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    st.success(f"Predicted Reach Score: **{pred:.2f}**")

# -------------------- TAB 3: DASHBOARD --------------------
with tab3:
    st.header("📊 Scholarship Dataset Dashboard")
    st.write("Explore and visualize the entire scholarship dataset.")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(sch_df, names='gender_norm', title="Distribution by Gender")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(sch_df['education_level'].value_counts().reset_index(),
                      x='index', y='education_level', title="Scholarships by Education Level")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💰 Income Limit Distribution")
    income_df = sch_df.dropna(subset=['income_limit_numeric'])
    fig3 = px.histogram(income_df, x='income_limit_numeric', nbins=15, title="Income Limit Histogram")
    st.plotly_chart(fig3, use_container_width=True)
