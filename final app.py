# final_app.py
# Finalized Streamlit app - Scholarship Finder + Reach Predictor + Dashboard
# Required files (place in same folder):
# - Curated_Scholarships_India_TN_200.csv
# - TN_Scholarship_Reach_REALISTIC.csv  (optional)
# - Linear_Regression_model.pkl etc. (optional)
# - scaler.pkl (optional)
# - anna-university-logo.png (optional)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# optional plotting
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTLY_AVAILABLE = False

# UI background (soft blue)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #f4f8fb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Anna University Scholarship App",
                   page_icon="🎓", layout="wide")

# Header
logo_path = "anna-university-logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=140)
st.title("🎓 Anna University Scholarship App")
st.markdown("**Eligibility Finder • Reach Predictor • Dataset Dashboard**")
st.markdown("---")

# ---------------- utilities ----------------
def safe_read_csv(path):
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}")
        st.stop()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed reading {path}: {e}")
        st.stop()

def detect_and_rename_scholarship_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common variants of column names in the uploaded CSV to standardized column names.
    If a standard name doesn't exist, create an empty column so downstream code won't fail.
    """
    orig_cols = list(df.columns)
    # map lowercase normalized names -> original name
    colmap = {c.lower().strip().replace(" ", "_"): c for c in orig_cols}

    def find_col(*keywords):
        for k in keywords:
            for low, orig in colmap.items():
                if k in low:
                    return orig
        return None

    mapping = {}
    mapping['scholarship_name'] = find_col("scholarship", "title", "name")
    mapping['level'] = find_col("level", "type", "scheme", "provider_type")
    mapping['category'] = find_col("category", "community", "caste")
    mapping['gender'] = find_col("gender", "sex")
    mapping['education_level'] = find_col("education", "eligible_classes", "eligible", "level")
    mapping['income_limit'] = find_col("income_limit", "income", "income_limit_numeric")
    mapping['amount'] = find_col("amount", "scholarship_amount", "value")
    mapping['website'] = find_col("website", "link", "url", "application_link", "official_website")
    mapping['description'] = find_col("description", "details", "note")
    mapping['provider'] = find_col("provider", "agency", "organisation", "organization")

    rename_dict = {}
    for std_name, orig in mapping.items():
        if orig and orig in df.columns:
            rename_dict[orig] = std_name
        else:
            df[std_name] = ""

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # ensure all standard columns exist
    for c in ['scholarship_name','level','category','gender','education_level','income_limit','amount','website','description','provider']:
        if c not in df.columns:
            df[c] = ""

    # trim whitespace
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    # numeric conversions
    try:
        if 'income_limit_numeric' not in df.columns:
            df['income_limit_numeric'] = df['income_limit'].astype(str).str.replace(r'[^\d.]','',regex=True).replace('', np.nan).astype(float)
        else:
            df['income_limit_numeric'] = pd.to_numeric(df['income_limit_numeric'], errors='coerce')
    except Exception:
        df['income_limit_numeric'] = np.nan

    try:
        if 'amount_numeric' not in df.columns:
            df['amount_numeric'] = df['amount'].astype(str).str.replace(r'[^\d.]','',regex=True).replace('', np.nan).astype(float)
        else:
            df['amount_numeric'] = pd.to_numeric(df['amount_numeric'], errors='coerce')
    except Exception:
        df['amount_numeric'] = np.nan

    return df

# Normalizers: handle many variants
def norm_gender(val):
    v = str(val).strip().lower()
    if v in ["", "nan", "na", "n/a", "all"]:
        return "All"
    if v.startswith("m") or v in ["male","man","boy","m"]:
        return "Male"
    if v.startswith("f") or v in ["female","woman","girl","f"]:
        return "Female"
    return "All"

def norm_category(val):
    v = str(val).strip().lower()
    if v in ["", "nan", "na", "n/a", "all"]:
        return "All"
    # common patterns
    if any(tok in v for tok in ["sc", "scheduled caste", "adi dravidar"]):
        return "SC"
    if any(tok in v for tok in ["st", "scheduled tribe"]):
        return "ST"
    if any(tok in v for tok in ["obc", "bc", "backward"]):
        return "OBC"
    if "minority" in v:
        return "Minority"
    if any(tok in v for tok in ["general", "open", "all community"]):
        return "General"
    # fallback uppercase
    return v.upper()

def norm_education(val):
    v = str(val).strip().lower()
    if v in ["", "nan", "na", "n/a", "all"]:
        return "All"
    if any(tok in v for tok in ["ug","undergrad","bachelor","b.tech","bsc","ba","be","b.e","b.com"]):
        return "Undergraduate"
    if any(tok in v for tok in ["pg","postgrad","master","m.tech","msc","ma","m.com","m.e"]):
        return "Postgraduate"
    if any(tok in v for tok in ["phd","doctor","doctoral"]):
        return "PhD"
    if any(tok in v for tok in ["school","higher secondary","hsc","12th","10th","secondary"]):
        return "School"
    return v.title()

def ensure_normalized_columns(df):
    # add columns if missing and normalize
    if 'gender_norm' not in df.columns:
        df['gender_norm'] = df['gender'].apply(norm_gender)
    else:
        df['gender_norm'] = df['gender'].fillna("").apply(norm_gender)

    if 'category_norm' not in df.columns:
        df['category_norm'] = df['category'].apply(norm_category)
    else:
        df['category_norm'] = df['category'].fillna("").apply(norm_category)

    if 'education_norm' not in df.columns:
        df['education_norm'] = df['education_level'].apply(norm_education)
    else:
        df['education_norm'] = df['education_level'].fillna("").apply(norm_education)

    df['gender_norm'] = df['gender_norm'].fillna("All")
    df['category_norm'] = df['category_norm'].fillna("All")
    df['education_norm'] = df['education_norm'].fillna("All")

    return df

# ---------------- load scholarship dataset ----------------
@st.cache_data
def load_scholarship_df(path="Curated_Scholarships_India_TN_200.csv"):
    df = safe_read_csv(path)
    df.columns = df.columns.str.strip()
    df = detect_and_rename_scholarship_columns(df)
    df = ensure_normalized_columns(df)
    return df

sch_df = load_scholarship_df()

# ---------------- load reach dataset (optional) ----------------
@st.cache_data
def load_reach_df(path="TN_Scholarship_Reach_REALISTIC.csv"):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'avg_family_income' in df.columns and 'school_infrastructure_index' in df.columns:
        df['income_to_infra'] = df['avg_family_income'] / df['school_infrastructure_index'].replace(0,1)
    if 'literacy_rate' in df.columns and 'schools_with_internet_percent' in df.columns:
        df['awareness_index'] = (df['literacy_rate'] * df['schools_with_internet_percent'])/100
    if 'district' not in df.columns:
        for alt in ['district_name','dist','region','area']:
            if alt in df.columns:
                df = df.rename(columns={alt:'district'})
                break
    if 'district' not in df.columns:
        df['district'] = "Unknown"
    return df

reach_df = load_reach_df()

# ---------------- load models & scaler (optional) ----------------
@st.cache_resource
def load_models_and_scaler():
    models = {}
    warnings = []
    for name, fname in [
        ("Linear Regression","Linear_Regression_model.pkl"),
        ("Random Forest","Random_Forest_model.pkl"),
        ("Gradient Boosting","Gradient_Boosting_model.pkl"),
    ]:
        if os.path.exists(fname):
            try:
                try:
                    models[name] = joblib.load(fname)
                except Exception:
                    with open(fname,'rb') as f:
                        models[name] = pickle.load(f)
            except Exception as e:
                warnings.append(f"Could not load {fname}: {e}")
        else:
            warnings.append(f"Model not found: {fname}")
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

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🏆 Eligibility Finder","📊 Reach Predictor","📈 Data Dashboard"])

# ---------------- TAB 1: Eligibility Finder ----------------
with tab1:
    st.header("🎯 Scholarship Eligibility Finder")
    st.markdown("Enter your details and click **Find Eligible Scholarships**. Results are shown as counts, cards (TN first) and a table. Click **Apply** to open the scholarship website.")

    # fixed simple input options
    genders = ["All","Male","Female"]
    categories = ["All","SC","ST","OBC","General","Minority"]
    edulevels = ["All","School","Undergraduate","Postgraduate","PhD"]

    col1, col2, col3 = st.columns(3)
    with col1:
        input_gender = st.selectbox("Gender", genders, index=0)
        input_category = st.selectbox("Category", categories, index=0)
    with col2:
        input_income = st.number_input("Annual Family Income (₹)", min_value=0, value=150000, step=5000)
        input_edu = st.selectbox("Education Level", edulevels, index=0)
    with col3:
        search_term = st.text_input("Search name or keyword", "")

    def filter_eligible(df):
        df2 = df.copy()
        df2 = ensure_normalized_columns(df2)

        # income filter
        if 'income_limit_numeric' in df2.columns:
            df2 = df2[(df2['income_limit_numeric'].isna()) | (input_income <= df2['income_limit_numeric'])]

        # gender filter using normalized column
        if input_gender != "All":
            df2 = df2[df2['gender_norm'].isin([input_gender, "All"])]

        # category: SC/ST should include Minority rows (as requested)
        if input_category != "All":
            if input_category in ["SC","ST"]:
                df2 = df2[df2['category_norm'].isin([input_category,"Minority","All"])]
            else:
                df2 = df2[df2['category_norm'].isin([input_category,"All"])]

        # education
        if input_edu != "All":
            df2 = df2[df2['education_norm'].isin([input_edu,"All"])]

        # search term
        if search_term.strip() != "":
            df2 = df2[
                df2['scholarship_name'].str.contains(search_term, case=False, na=False) |
                df2['description'].str.contains(search_term, case=False, na=False)
            ]

        return df2

    if st.button("🔎 Find Eligible Scholarships"):
        eligible_df = filter_eligible(sch_df)
        st.session_state['eligible_df'] = eligible_df  # save for dashboard option

        if eligible_df.empty:
            st.warning("No scholarships matched your criteria. Try relaxing filters.")
        else:
            # counts and metrics
            tn_mask = eligible_df['level'].fillna("").str.contains("state|tn|tamil", case=False, na=False)
            central_mask = eligible_df['level'].fillna("").str.contains("central|national", case=False, na=False)
            tn_df = eligible_df[tn_mask]
            central_df = eligible_df[central_mask]
            c1, c2, c3 = st.columns(3)
            c1.metric("🏛️ Tamil Nadu", len(tn_df))
            c2.metric("🇮🇳 Central", len(central_df))
            c3.metric("📄 Total", len(eligible_df))

            st.markdown("---")
            view_mode = st.radio("View as", ["Cards","Table"], horizontal=True)

            if view_mode == "Table":
                cols = ['scholarship_name','level','category','gender','education_level','income_limit','amount','website']
                cols = [c for c in cols if c in eligible_df.columns]
                table_df = eligible_df[cols].rename(columns={'scholarship_name':'Scholarship Name','education_level':'Education Level','income_limit':'Income Limit'})
                st.dataframe(table_df.reset_index(drop=True), use_container_width=True)
                # download
                csv = table_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results (CSV)", csv, file_name="eligible_scholarships.csv", mime="text/csv")
            else:
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
                        if web and web.lower() and not web.lower().startswith("http"):
                            web = "https://" + web
                        st.markdown(
                            f"""
                            <div style='background:linear-gradient(90deg,#eefaf6,#e6f0ff);padding:12px;border-radius:10px;margin-bottom:10px;'>
                              <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <div style='max-width:78%;'>
                                  <h4 style='margin:0;color:#0b5ed7'>{name}</h4>
                                  <small style='color:#333'>{prov}</small>
                                  <p style='margin:6px 0 0 0;'><b>Category:</b> {cat} &nbsp; | &nbsp; <b>Gender:</b> {gen} &nbsp; | &nbsp; <b>Edu:</b> {edu}</p>
                                  <p style='margin:6px 0 0 0;'><b>Income Limit:</b> {inc} &nbsp; | &nbsp; <b>Amount:</b> {amt}</p>
                                </div>
                                <div style='text-align:right;'>
                                  <a href="{web}" target="_blank" style='background:#198754;color:white;padding:8px 12px;border-radius:8px;text-decoration:none;'>Apply</a>
                                </div>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No Tamil Nadu scholarships found for these inputs.")

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
                        if web and web.lower() and not web.lower().startswith("http"):
                            web = "https://" + web
                        st.markdown(
                            f"""
                            <div style='background:linear-gradient(90deg,#fff8f0,#f3f0ff);padding:12px;border-radius:10px;margin-bottom:10px;'>
                              <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <div style='max-width:78%;'>
                                  <h4 style='margin:0;color:#6f42c1'>{name}</h4>
                                  <small style='color:#333'>{prov}</small>
                                  <p style='margin:6px 0 0 0;'><b>Category:</b> {cat} &nbsp; | &nbsp; <b>Gender:</b> {gen} &nbsp; | &nbsp; <b>Edu:</b> {edu}</p>
                                  <p style='margin:6px 0 0 0;'><b>Income Limit:</b> {inc} &nbsp; | &nbsp; <b>Amount:</b> {amt}</p>
                                </div>
                                <div style='text-align:right;'>
                                  <a href="{web}" target="_blank" style='background:#0d6efd;color:white;padding:8px 12px;border-radius:8px;text-decoration:none;'>Apply</a>
                                </div>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No Central scholarships found for these inputs.")

# ---------------- TAB 2: Reach Predictor ----------------
with tab2:
    st.header("📊 Scholarship Reach Predictor (District-level)")
    if reach_df.empty:
        st.warning("Reach dataset (TN_Scholarship_Reach_REALISTIC.csv) not found or empty. This tab requires that CSV to give district-level defaults.")
    else:
        # derived features
        if 'income_to_infra' not in reach_df.columns and 'avg_family_income' in reach_df.columns and 'school_infrastructure_index' in reach_df.columns:
            reach_df['income_to_infra'] = reach_df['avg_family_income'] / reach_df['school_infrastructure_index'].replace(0,1)
        if 'awareness_index' not in reach_df.columns and 'literacy_rate' in reach_df.columns and 'schools_with_internet_percent' in reach_df.columns:
            reach_df['awareness_index'] = (reach_df['literacy_rate'] * reach_df['schools_with_internet_percent'])/100

        required_feats = [
            "avg_family_income","literacy_rate","female_ratio","rural_population_percent",
            "num_students","schools_with_computer_lab_percent","schools_with_internet_percent",
            "school_infrastructure_index","income_to_infra","awareness_index"
        ]
        missing = [f for f in required_feats if f not in reach_df.columns]
        if missing:
            st.warning(f"Warning: reach dataset missing expected cols: {missing}")

        if models:
            model_choice = st.selectbox("Choose model", list(models.keys()))
        else:
            st.error("No models loaded (optional). Prediction needs model .pkl files.")
            model_choice = None

        district_list = list(reach_df['district'].dropna().unique())
        district = st.selectbox("Select District", district_list)
        row = reach_df[reach_df['district']==district].iloc[0] if district in reach_df['district'].values else reach_df.iloc[0]

        # inputs
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

        income_to_infra = avg_income / (infra_idx if infra_idx!=0 else 1)
        awareness_index = (literacy * internet_pct)/100

        feat_vector = [avg_income, literacy, female_ratio, rural_pct, num_students, comp_lab, internet_pct, infra_idx, income_to_infra, awareness_index]
        X = np.array([feat_vector])

        if scaler is not None:
            try:
                Xs = scaler.transform(X)
            except Exception as e:
                st.warning(f"Scaler transform failed ({e}), using raw features.")
                Xs = X
        else:
            Xs = X

        if st.button("🚀 Predict Reach"):
            if not models:
                st.error("No models available.")
            else:
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
    st.markdown("Visualize either the full dataset or the last eligible search results (if you ran Tab 1). Use filters to drill down.")

    # choose data source
    use_eligible = st.checkbox("Visualize last eligible search results (if available)", value=False)
    if use_eligible and st.session_state.get('eligible_df') is not None:
        df_vis = st.session_state['eligible_df'].copy()
        df_vis = ensure_normalized_columns(df_vis)
        st.info("Showing last eligible search results.")
    else:
        df_vis = sch_df.copy()

    # filters
    col1, col2, col3 = st.columns(3)
    with col1:
        level_options = sorted(df_vis['level'].dropna().unique()) if 'level' in df_vis.columns else ["All"]
        level_filter = st.multiselect("Level", options=level_options, default=level_options)
        cat_filter = st.multiselect("Category", options=["All","SC","ST","OBC","General","Minority"], default=["All"])
    with col2:
        gender_filter = st.multiselect("Gender", options=["All","Male","Female"], default=["All"])
        edu_filter = st.multiselect("Education", options=["All","School","Undergraduate","Postgraduate","PhD"], default=["All"])
    with col3:
        safe_max = int(df_vis['income_limit_numeric'].max(skipna=True)) if 'income_limit_numeric' in df_vis.columns and pd.notna(df_vis['income_limit_numeric'].max()) else 1000000
        income_range = st.slider("Income limit range (₹)", 0, max(1000000,safe_max), (0, max(1000000,safe_max)))
        show_top = st.slider("Top N categories", min_value=3, max_value=30, value=10)

    # apply filters
    dff = df_vis.copy()
    # category
    if "All" not in cat_filter:
        mask_cat = pd.Series(False, index=dff.index)
        for sel in cat_filter:
            if sel in ["SC","ST"]:
                mask_cat = mask_cat | dff['category_norm'].isin([sel,"Minority","All"])
            else:
                mask_cat = mask_cat | dff['category_norm'].isin([sel,"All"])
        dff = dff[mask_cat]

    # gender
    if "All" not in gender_filter:
        dff = dff[dff['gender_norm'].isin(gender_filter) | (dff['gender_norm']=="All")]

    # education
    if "All" not in edu_filter:
        dff = dff[dff['education_norm'].isin(edu_filter) | (dff['education_norm']=="All")]

    # level
    if level_filter:
        dff = dff[dff['level'].isin(level_filter)]

    # income numeric
    if 'income_limit_numeric' in dff.columns:
        dff = dff[
            (dff['income_limit_numeric'].fillna(-1) >= income_range[0]) &
            (dff['income_limit_numeric'].fillna(1e12) <= income_range[1])
        ]

    # metrics
    t1, t2, t3 = st.columns(3)
    t1.metric("Total Scholarships", len(dff))
    t2.metric("Tamil Nadu (approx)", len(dff[dff['level'].fillna('').str.contains("state|tn|tamil", case=False, na=False)]))
    t3.metric("Central (approx)", len(dff[dff['level'].fillna('').str.contains("central|national", case=False, na=False)]))

    st.markdown("---")
    # charts
    if not dff.empty and PLOTLY_AVAILABLE:
        st.subheader("Top categories")
        vc = dff['category_norm'].value_counts().nlargest(show_top).reset_index()
        vc.columns = ['Category','Count']
        fig = px.bar(vc, x='Category', y='Count', title=f"Top {show_top} Categories")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Level distribution")
        if 'level' in dff.columns:
            lvl = dff['level'].value_counts().reset_index()
            lvl.columns = ['Level','Count']
            fig2 = px.pie(lvl, values='Count', names='Level', title="Level distribution")
            st.plotly_chart(fig2, use_container_width=True)

        # amount histogram
        if 'amount_numeric' in dff.columns and dff['amount_numeric'].notna().any():
            st.subheader("Amount distribution")
            fig3 = px.histogram(dff, x='amount_numeric', nbins=30, title='Distribution of scholarship amounts')
            st.plotly_chart(fig3, use_container_width=True)
    elif not PLOTLY_AVAILABLE:
        st.info("Plotly not available. Charts will be basic.")

    st.markdown("---")
    st.subheader("Filtered Data")
    st.dataframe(dff.reset_index(drop=True), use_container_width=True)

# footer
st.markdown("---")
st.markdown("**Developed by:** Logesh Kannan  ·  **Guide:** Dr. Rajkumar  · Anna University Regional Campus, Madurai")

