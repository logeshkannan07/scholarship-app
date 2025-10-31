# final_app.py
# Robust Streamlit app for Scholarship Finder + Reach Predictor + Dashboard
# Put this file in the same folder as:
# - Curated_Scholarships_India_TN_200.csv
# - TN_Scholarship_Reach_REALISTIC.csv
# - Linear_Regression_model.pkl
# - Random_Forest_model.pkl
# - Gradient_Boosting_model.pkl
# - scaler.pkl
# - anna-university-logo.png (optional)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import plotly.express as px
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #fff8f8;  /* Soft light blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# plotting libs (try plotly, fallback to matplotlib)
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTLY_AVAILABLE = False

# ---------------- App Config ----------------
st.set_page_config(page_title="Anna University Scholarship App",
                   page_icon="🎓", layout="wide")

# Header with logo (if present)
logo_path = "anna-university-logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=140)
st.title("🎓 Anna University Scholarship App")
st.markdown("**Eligibility Finder • Reach Predictor • Dataset Dashboard**")
st.markdown("---")

# ---------------- Helpers: robust column detection ----------------
def detect_and_rename_scholarship_columns(df: pd.DataFrame) -> pd.DataFrame:
    # make a lowercase map for detection
    orig_cols = list(df.columns)
    colmap = {c: c.lower().strip().replace(" ", "_") for c in orig_cols}

    # helper to find column by keyword possibilities
    def find_col(*keywords):
        for k in keywords:
            for orig, low in colmap.items():
                if k in low:
                    return orig
        return None

    mapping = {}
    mapping['scholarship_name'] = find_col("scholarship", "name", "title")
    mapping['level'] = find_col("level", "type", "scheme")
    mapping['category'] = find_col("category", "community", "caste")
    mapping['gender'] = find_col("gender", "sex")
    mapping['education_level'] = find_col("education", "eligible_classes", "eligible")
    mapping['income_limit'] = find_col("income_limit", "income", "income_limit")
    mapping['amount'] = find_col("amount", "scholarship_amount", "value")
    mapping['website'] = find_col("website", "link", "url", "application_link")
    mapping['description'] = find_col("description", "details", "note")
    mapping['provider'] = find_col("provider", "agency", "organisation", "organization")

    # build rename dict for pandas
    rename_dict = {}
    for std_name, orig in mapping.items():
        if orig and orig in df.columns:
            rename_dict[orig] = std_name
        else:
            # create blank column if not found
            df[std_name] = ""
    if rename_dict:
        df = df.rename(columns=rename_dict)
    # final ensure standard columns exist
    for c in ['scholarship_name','level','category','gender','education_level','income_limit','amount','website','description','provider']:
        if c not in df.columns:
            df[c] = ""
    # strip spaces in string columns
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    # numeric conversions
    try:
        df['income_limit_numeric'] = df['income_limit'].astype(str).str.replace(r'[^\d.]','',regex=True).replace('', np.nan).astype(float)
    except Exception:
        df['income_limit_numeric'] = np.nan
    try:
        df['amount_numeric'] = df['amount'].astype(str).str.replace(r'[^\d.]','',regex=True).replace('', np.nan).astype(float)
    except Exception:
        df['amount_numeric'] = np.nan
    return df

def safe_read_csv(path):
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}")
        st.stop()
    return pd.read_csv(path)

# ---------------- Load scholarship dataset ----------------
@st.cache_data
def load_scholarship_df(path="Curated_Scholarships_India_TN_200.csv"):
    df = safe_read_csv(path)
    df.columns = df.columns.str.strip()
    df = detect_and_rename_scholarship_columns(df)
    return df

sch_df = load_scholarship_df()

# ---------------- Load reach dataset ----------------
@st.cache_data
def load_reach_df(path="TN_Scholarship_Reach_REALISTIC.csv"):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # compute derived if possible
    if 'avg_family_income' in df.columns and 'school_infrastructure_index' in df.columns:
        df['income_to_infra'] = df['avg_family_income'] / df['school_infrastructure_index'].replace(0,1)
    if 'literacy_rate' in df.columns and 'schools_with_internet_percent' in df.columns:
        df['awareness_index'] = (df['literacy_rate'] * df['schools_with_internet_percent'])/100
    return df

reach_df = load_reach_df()

# ---------------- Safe model/scaler loading ----------------
@st.cache_resource
def load_models_and_scaler():
    models = {}
    warnings = []
    # attempt to load each model
    for name, fname in [
        ("Linear Regression","Linear_Regression_model.pkl"),
        ("Random Forest","Random_Forest_model.pkl"),
        ("Gradient Boosting","Gradient_Boosting_model.pkl"),
    ]:
        if os.path.exists(fname):
            try:
                # prefer joblib if pickled via joblib
                try:
                    models[name] = joblib.load(fname)
                except Exception:
                    with open(fname,'rb') as f:
                        models[name] = pickle.load(f)
                # success
            except Exception as e:
                warnings.append(f"Could not load {fname}: {e}")
        else:
            warnings.append(f"Model file not found: {fname}")
    # load scaler safe
    scaler = None
    if os.path.exists("scaler.pkl"):
        try:
            try:
                scaler = joblib.load("scaler.pkl")
            except Exception:
                with open("scaler.pkl","rb") as f:
                    scaler = pickle.load(f)
        except Exception as e:
            warnings.append(f"Scaler load failed: {e}")
            scaler = None
    else:
        warnings.append("scaler.pkl not found.")
    return models, scaler, warnings

models, scaler, load_warnings = load_models_and_scaler()
for w in load_warnings:
    st.warning(w)

# ---------------- UI tabs ----------------
tab1, tab2, tab3 = st.tabs(["🏆 Eligibility Finder","📊 Reach Predictor","📈 Data Dashboard"])

# ---------------- TAB 1: Eligibility Finder (upgraded cards) ----------------
with tab1:
    st.header("🎯 Scholarship Eligibility Finder")
    st.markdown("Enter your details to find eligible scholarships. Results displayed as clickable cards (Tamil Nadu / Central).")

    # controls with safe options
    genders = ["All"] + sorted(set([g for g in sch_df['gender'].unique() if g and g.lower()!="nan"]))
    categories = ["All"] + sorted(set([c for c in sch_df['category'].unique() if c and c.lower()!="nan"]))
    edulevels = ["All"] + sorted(set([e for e in sch_df['education_level'].unique() if e and e.lower()!="nan"]))

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
        # income: use income_limit_numeric if present
        if 'income_limit_numeric' in df2.columns:
            # keep rows where income limit is NaN OR input_income <= limit
            df2 = df2[(df2['income_limit_numeric'].isna()) | (input_income <= df2['income_limit_numeric'])]
        # gender
        if input_gender and input_gender != "All":
            df2 = df2[(df2['gender'].str.lower() == input_gender.lower()) | (df2['gender'].str.lower() == "all")]
        # category
        if input_category and input_category != "All":
            df2 = df2[(df2['category'].str.upper() == input_category.upper()) | (df2['category'].str.upper() == "ALL")]
        # education
        if input_edu and input_edu != "All":
            df2 = df2[df2['education_level'].str.contains(input_edu, case=False, na=False) | (df2['education_level'].str.lower()=="all")]
        # search
        if search_term:
            df2 = df2[df2['scholarship_name'].str.contains(search_term, case=False, na=False) | df2['description'].str.contains(search_term, case=False, na=False)]
        return df2

    if st.button("🔎 Find Eligible Scholarships"):
        eligible_df = filter_eligible(sch_df)
        st.session_state['eligible_df'] = eligible_df  # store for dashboard if needed

        if eligible_df.empty:
            st.warning("No scholarships matched your criteria. Try loosening filters.")
        else:
            st.success(f"Found {len(eligible_df)} scholarships.")
            # counts
            tn_df = eligible_df[eligible_df['level'].str.contains("state|tn|tamil", case=False, na=False)]
            central_df = eligible_df[eligible_df['level'].str.contains("central|national", case=False, na=False)]
            c1, c2, c3 = st.columns(3)
            c1.metric("🏛️ Tamil Nadu", len(tn_df))
            c2.metric("🇮🇳 Central", len(central_df))
            c3.metric("📄 Total", len(eligible_df))

            st.markdown("---")
            view_mode = st.radio("View as", ["Cards","Table"], horizontal=True)

            if view_mode == "Table":
                show_cols = ['scholarship_name','level','category','gender','education_level','income_limit','amount','website']
                st.dataframe(eligible_df[show_cols].rename(columns={
                    'scholarship_name':'Scholarship Name','education_level':'Education Level','income_limit':'Income Limit'}).reset_index(drop=True), use_container_width=True)
            else:
                # cards: TN first then Central
                st.markdown("### 🟢 Tamil Nadu Scholarships")
                if not tn_df.empty:
                    for _, r in tn_df.iterrows():
                        name = r.get('scholarship_name','')
                        prov = r.get('provider','')
                        cat = r.get('category','')
                        gen = r.get('gender','')
                        edu = r.get('education_level','')
                        inc = r.get('income_limit','')
                        amt = r.get('amount','')
                        web = r.get('website','').strip()
                        if web and not web.lower().startswith("http"):
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
                        name = r.get('scholarship_name','')
                        prov = r.get('provider','')
                        cat = r.get('category','')
                        gen = r.get('gender','')
                        edu = r.get('education_level','')
                        inc = r.get('income_limit','')
                        amt = r.get('amount','')
                        web = r.get('website','').strip()
                        if web and not web.lower().startswith("http"):
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

# ---------------- TAB 2: Reach Predictor (fixed features) ----------------
with tab2:
    st.header("📊 Scholarship Reach Predictor (District-level)")

    if reach_df.empty:
        st.warning("Reach dataset TN_Scholarship_Reach_REALISTIC.csv not found or empty.")
    else:
        # ensure derived fields exist
        if 'income_to_infra' not in reach_df.columns and 'avg_family_income' in reach_df.columns and 'school_infrastructure_index' in reach_df.columns:
            reach_df['income_to_infra'] = reach_df['avg_family_income'] / reach_df['school_infrastructure_index'].replace(0,1)
        if 'awareness_index' not in reach_df.columns and 'literacy_rate' in reach_df.columns and 'schools_with_internet_percent' in reach_df.columns:
            reach_df['awareness_index'] = (reach_df['literacy_rate'] * reach_df['schools_with_internet_percent'])/100

        required_feats = [
            "avg_family_income","literacy_rate","female_ratio","rural_population_percent",
            "num_students","schools_with_computer_lab_percent","schools_with_internet_percent",
            "school_infrastructure_index","income_to_infra","awareness_index"
        ]
        # check available features
        available_feats = [f for f in required_feats if f in reach_df.columns]
        missing = [f for f in required_feats if f not in reach_df.columns]
        if missing:
            st.warning(f"Some expected features are missing from reach dataset: {missing}. Prediction may fail or be less accurate.")

        # model selector
        if models:
            model_choice = st.selectbox("Choose model", list(models.keys()))
        else:
            st.error("No models loaded. Upload model .pkl files.")
            st.stop()

        district = st.selectbox("Select District", reach_df["district"].unique())
        row = reach_df[reach_df["district"]==district].iloc[0]

        # populate inputs with district defaults
        colA, colB = st.columns(2)
        with colA:
            avg_income = st.number_input("Average Family Income", value=float(row.get('avg_family_income',0)))
            literacy = st.number_input("Literacy Rate (%)", value=float(row.get('literacy_rate',0)))
            female_ratio = st.number_input("Female Ratio", value=float(row.get('female_ratio',0)))
            rural_pct = st.number_input("Rural Population (%)", value=float(row.get('rural_population_percent',0)))
            num_students = st.number_input("Number of Students", value=int(row.get('num_students',0)))
        with colB:
            comp_lab = st.number_input("Schools with Computer Lab (%)", value=float(row.get('schools_with_computer_lab_percent',0)))
            internet_pct = st.number_input("Schools with Internet (%)", value=float(row.get('schools_with_internet_percent',0)))
            infra_idx = st.number_input("School Infrastructure Index", value=float(row.get('school_infrastructure_index',0)))

        # derived
        income_to_infra = avg_income / (infra_idx if infra_idx!=0 else 1)
        awareness_index = (literacy * internet_pct)/100

        # build feature vector in exact order
        feat_vector = [avg_income, literacy, female_ratio, rural_pct, num_students, comp_lab, internet_pct, infra_idx, income_to_infra, awareness_index]
        X = np.array([feat_vector])

        # scale if possible
        if scaler is not None:
            try:
                Xs = scaler.transform(X)
            except Exception as e:
                st.warning(f"Scaler transform failed ({e}). Using raw features.")
                Xs = X
        else:
            Xs = X

        if st.button("🚀 Predict Reach"):
            model = models.get(model_choice)
            if model is None:
                st.error("Selected model not loaded.")
            else:
                try:
                    pred = model.predict(Xs)[0]
                    st.success(f"🎯 Predicted Scholarship Reach in {district}: {pred:.2f}%")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ---------------- TAB 3: Data Dashboard ----------------
with tab3:
    st.header("📈 Dataset Visualization Dashboard")
    st.markdown("Explore counts, distributions and amounts. Use filters to focus on particular segments.")

    # choose source: last eligible set if exists else full
    base_df = st.session_state.get('eligible_df', sch_df)

    # filters
    col1, col2, col3 = st.columns(3)
    with col1:
        level_filter = st.multiselect("Filter by Level", options=sorted(base_df['level'].dropna().unique()), default=sorted(base_df['level'].dropna().unique()))
    with col2:
        cat_filter = st.multiselect("Filter by Category", options=sorted(base_df['category'].dropna().unique()), default=sorted(base_df['category'].dropna().unique()))
    with col3:
        show_top = st.slider("Top N categories (for bar chart)", min_value=3, max_value=30, value=10)

    df_vis = base_df[(base_df['level'].isin(level_filter)) & (base_df['category'].isin(cat_filter))]

    st.subheader("Summary Metrics")
    t1, t2, t3 = st.columns(3)
    t1.metric("Total Scholarships", len(df_vis))
    t2.metric("Tamil Nadu (approx)", len(df_vis[df_vis['level'].str.contains("state|tn|tamil", case=False, na=False)]))
    t3.metric("Central (approx)", len(df_vis[df_vis['level'].str.contains("central|national", case=False, na=False)]))

    st.markdown("---")
    # charts
    if PLOTLY_AVAILABLE:
        st.subheader("Bar chart — top categories")
        vc = df_vis['category'].value_counts().nlargest(show_top).reset_index()
        vc.columns = ['Category','Count']
        fig = px.bar(vc, x='Category', y='Count', title=f"Top {show_top} Categories")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pie — Level distribution")
        lvl = df_vis['level'].value_counts().reset_index()
        lvl.columns = ['Level','Count']
        fig2 = px.pie(lvl, values='Count', names='Level', title="Level distribution")
        st.plotly_chart(fig2, use_container_width=True)

        # amount histogram
        if 'amount_numeric' in df_vis.columns and df_vis['amount_numeric'].notna().any():
            st.subheader("Amount distribution")
            fig3 = px.histogram(df_vis, x='amount_numeric', nbins=30, title='Distribution of Amounts')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Amount values not numeric or missing; histogram not shown.")
    else:
        st.info("Plotly not available — showing Matplotlib charts")
        # bar
        vc = df_vis['category'].value_counts().nlargest(show_top)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_title(f"Top {show_top} Categories")
        st.pyplot(fig)
        # pie
        lvl = df_vis['level'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(lvl.values, labels=lvl.index, autopct="%1.1f%%")
        ax2.axis('equal')
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Data (filtered)")
    st.dataframe(df_vis.reset_index(drop=True), use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("**Developed by:** Logesh Kannan  ·  **Guide:** Dr. Rajkumar  · Anna University Regional Campus, Madurai")





