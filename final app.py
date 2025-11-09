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
# ---------------- TAB 1: Eligibility Finder (fixed logic) ----------------
with tab1:
    st.header("🎯 Scholarship Eligibility Finder")
    st.markdown("Enter your details to find eligible scholarships. Results displayed as clickable cards (Tamil Nadu / Central).")

    # ----- Simplified fixed filter options (updated lists) -----
    genders = ["All", "Male", "Female", "Transgender"]
    categories = ["All", "OBC/BC", "SC", "ST", "General", "MBC"]
    edulevels = ["All", "UG", "PG", "PhD", "School"]

    col1, col2, col3 = st.columns(3)
    with col1:
        input_gender = st.selectbox("Gender", genders, index=0)
        input_category = st.selectbox("Category", categories, index=0)
    with col2:
        input_income = st.number_input("Annual Family Income (₹)", min_value=0, value=150000, step=5000)
        input_edu = st.selectbox("Education Level", edulevels, index=0)
    with col3:
        search_term = st.text_input("Search scholarship name or keyword", "")

    # filtering function (uses standardized columns and tags)
    def filter_eligible(df):
        df2 = df.copy()

        # Income filter
        if 'income_limit_numeric' in df2.columns:
            df2 = df2[(df2['income_limit_numeric'].isna()) | (input_income <= df2['income_limit_numeric'])]

        # Gender logic:
        # Male → exclude Female-only
        # Female → show Female and All
        # Transgender → show Transgender and All
        if input_gender != "All":
            if input_gender == "Male":
                df2 = df2[~df2['gender_norm'].eq('Female')]
            elif input_gender == "Female":
                df2 = df2[df2['gender_norm'].isin(['Female', 'All'])]
            elif input_gender == "Transgender":
                df2 = df2[df2['gender_norm'].isin(['Transgender', 'All'])]

        # Category logic (updated)
        if input_category != "All":
            sel = input_category.lower()
            def category_matches(tags):
                if not isinstance(tags, (set, list)):
                    return False
                t = set([str(x).lower() for x in tags])
                if sel in ['obc/bc', 'obc', 'bc']:
                    return ('obc' in t) or ('bc' in t)
                if sel == 'sc':
                    return ('sc' in t)
                if sel == 'st':
                    return ('st' in t)
                if sel == 'general':
                    return ('general' in t)
                if sel == 'mbc':
                    return ('mbc' in t)
                return any(sel in x for x in t)
            df2 = df2[df2['category_tags'].apply(category_matches)]

        # Education logic (updated UG/PG/PhD/School)
        if input_edu != "All":
            sel = input_edu.lower()
            def edu_matches(tags):
                if not isinstance(tags, (set, list)):
                    return False
                t = set([str(x).lower() for x in tags])
                if sel == 'ug':
                    return ('ug' in t) or ('ug/pg' in t)
                if sel == 'pg':
                    return ('pg' in t) or ('ug/pg' in t)
                if sel == 'phd':
                    return ('phd' in t)
                if sel == 'school':
                    return ('school' in t)
                if sel == 'all':
                    return True
                return False
            df2 = df2[df2['edu_tags'].apply(edu_matches)]

        # Search term
        if search_term:
            df2 = df2[
                df2['scholarship_name'].str.contains(search_term, case=False, na=False) |
                df2['description'].str.contains(search_term, case=False, na=False)
            ]

        return df2

    if st.button("🔎 Find Eligible Scholarships"):
        eligible_df = filter_eligible(sch_df)
        st.session_state['eligible_df'] = eligible_df

        if eligible_df.empty:
            st.warning("No scholarships matched your criteria. Try loosening filters.")
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
                show_cols = ['scholarship_name', 'level', 'category', 'gender', 'education_level', 'income_limit', 'amount', 'website']
                show_cols = [c for c in show_cols if c in eligible_df.columns]
                st.dataframe(
                    eligible_df[show_cols].rename(columns={
                        'scholarship_name': 'Scholarship Name',
                        'education_level': 'Education Level',
                        'income_limit': 'Income Limit'
                    }).reset_index(drop=True),
                    use_container_width=True
                )
            else:
                st.markdown("### 🟢 Tamil Nadu Scholarships")
                if not tn_df.empty:
                    for _, r in tn_df.iterrows():
                        name = r.get('scholarship_name', '')
                        prov = r.get('provider', '')
                        cat = r.get('category', '')
                        gen = r.get('gender', '')
                        edu = r.get('education_level', '')
                        inc = r.get('income_limit', '')
                        amt = r.get('amount', '')
                        web = r.get('website', '').strip()
                        if web and web.lower() and not web.lower().startswith("http"):
                            web = "https://" + web
                        st.markdown(f"""
                            <div style='background:linear-gradient(90deg,#f0fff4,#e6f7ff);padding:12px;border-radius:10px;margin-bottom:10px;'>
                              <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <div style='max-width:78%;'>
                                  <h4 style='margin:0;color:#0a58ca'>{name}</h4>
                                  <small style='color:#333'>{prov}</small>
                                  <p style='margin:6px 0 0 0;'><b>Category:</b> {cat} &nbsp; | &nbsp; <b>Gender:</b> {gen} &nbsp; | &nbsp; <b>Edu:</b> {edu}</p>
                                  <p style='margin:6px 0 0 0;'><b>Income Limit:</b> {inc} &nbsp; | &nbsp; <b>Amount:</b> {amt}</p>
                                </div>
                                <div style='text-align:right;'>
                                  <a href="{web}" target="_blank" style='background:#198754;color:white;padding:7px 12px;border-radius:8px;text-decoration:none;'>Apply</a>
                                </div>
                              </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No Tamil Nadu scholarships found.")

                st.markdown("### 🟣 Central Scholarships")
                if not central_df.empty:
                    for _, r in central_df.iterrows():
                        name = r.get('scholarship_name', '')
                        prov = r.get('provider', '')
                        cat = r.get('category', '')
                        gen = r.get('gender', '')
                        edu = r.get('education_level', '')
                        inc = r.get('income_limit', '')
                        amt = r.get('amount', '')
                        web = r.get('website', '').strip()
                        if web and web.lower() and not web.lower().startswith("http"):
                            web = "https://" + web
                        st.markdown(f"""
                            <div style='background:linear-gradient(90deg,#fff8f0,#f3f0ff);padding:12px;border-radius:10px;margin-bottom:10px;'>
                              <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <div style='max-width:78%;'>
                                  <h4 style='margin:0;color:#7b1fa2'>{name}</h4>
                                  <small style='color:#333'>{prov}</small>
                                  <p style='margin:6px 0 0 0;'><b>Category:</b> {cat} &nbsp; | &nbsp; <b>Gender:</b> {gen} &nbsp; | &nbsp; <b>Edu:</b> {edu}</p>
                                  <p style='margin:6px 0 0 0;'><b>Income Limit:</b> {inc} &nbsp; | &nbsp; <b>Amount:</b> {amt}</p>
                                </div>
                                <div style='text-align:right;'>
                                  <a href="{web}" target="_blank" style='background:#0d6efd;color:white;padding:7px 12px;border-radius:8px;text-decoration:none;'>Apply</a>
                                </div>
                              </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No Central scholarships found.")



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


