# final_app.py
# Robust Streamlit app — Eligibility Finder (cards + table), Reach Predictor (3 models), Dashboard
# Place this file with your CSVs and model .pkl files.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# plotting libs
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTLY = False

# ---------------- App config & styles ----------------
st.set_page_config(page_title="Anna University Scholarship App", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #ffffff; }
.card { padding: 12px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
.scholar-card { padding:12px; border-radius:10px; margin-bottom:10px; display:block; }
.tn { background: linear-gradient(90deg,#f0fff4,#e6f7ff); border-left:6px solid #198754; padding:12px; }
.central { background: linear-gradient(90deg,#fff8f0,#f3f0ff); border-left:6px solid #0d6efd; padding:12px; }
a.apply-btn { background:#0d6efd;color:white;padding:7px 12px;border-radius:8px;text-decoration:none;}
a.apply-btn-tn { background:#198754;color:white;padding:7px 12px;border-radius:8px;text-decoration:none;}
</style>
""", unsafe_allow_html=True)

# Header with logo if exists
logo = "anna-university-logo.png"
if os.path.exists(logo):
    st.image(logo, width=140)
st.title("🎓 Anna University Scholarship App")
st.markdown("**Eligibility Finder • Reach Predictor • Data Dashboard**")
st.markdown("---")

# ---------------- Helpers ----------------

def safe_read_csv(path):
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}")
        st.stop()
    return pd.read_csv(path)

def normalize_scholar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make robust standard columns available in dataframe. Returns df with standardized lower_snake_case columns where possible."""
    df = df.copy()
    # keep original columns
    orig = list(df.columns)
    lowered = {c: c.lower().strip().replace(" ", "_") for c in orig}

    def find_col(*keys):
        for k in keys:
            for orig_c, low in lowered.items():
                if k in low:
                    return orig_c
        return None

    mapping = {
        "scholarship_name": find_col("scholarship", "name", "title"),
        "level": find_col("level", "type", "scheme"),
        "category": find_col("category", "caste", "community"),
        "gender": find_col("gender", "sex"),
        "education_level": find_col("education", "eligible_classes", "eligible"),
        "income_limit": find_col("income_limit", "income", "income_limit_(₹)"),
        "amount": find_col("amount", "scholarship_amount", "value"),
        "website": find_col("website", "link", "url", "application_link"),
        "description": find_col("description", "details", "note"),
        "provider": find_col("provider", "agency", "organisation", "organization")
    }

    rename_dict = {}
    for std, orig_name in mapping.items():
        if orig_name and orig_name in df.columns:
            rename_dict[orig_name] = std
        else:
            # create blank column to avoid KeyError later
            df[std] = ""

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # ensure final columns exist
    for c in ["scholarship_name","level","category","gender","education_level","income_limit","amount","website","description","provider"]:
        if c not in df.columns:
            df[c] = ""

    # strip whitespace from string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # numeric conversion attempts
    try:
        df["income_limit_numeric"] = df["income_limit"].astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", np.nan).astype(float)
    except Exception:
        df["income_limit_numeric"] = np.nan
    try:
        df["amount_numeric"] = df["amount"].astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", np.nan).astype(float)
    except Exception:
        df["amount_numeric"] = np.nan

    # standardize level strings
    df["level"] = df["level"].astype(str)
    return df

@st.cache_data
def load_scholarship_df(path="Curated_Scholarships_India_TN_200.csv"):
    df = safe_read_csv(path)
    df.columns = df.columns.str.strip()
    df = normalize_scholar_columns(df)
    return df

@st.cache_data
def load_reach_df(path="TN_Scholarship_Reach_REALISTIC.csv"):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # add derived features if possible
    if "avg_family_income" in df.columns and "school_infrastructure_index" in df.columns:
        df["income_to_infra"] = df["avg_family_income"] / df["school_infrastructure_index"].replace(0,1)
    if "literacy_rate" in df.columns and "schools_with_internet_percent" in df.columns:
        df["awareness_index"] = (df["literacy_rate"] * df["schools_with_internet_percent"]) / 100
    return df

@st.cache_resource
def load_models_and_scaler():
    models = {}
    warnings = []
    model_files = {
        "Linear Regression": "Linear_Regression_model.pkl",
        "Random Forest": "Random_Forest_model.pkl",
        "Gradient Boosting": "Gradient_Boosting_model.pkl"
    }
    for name, fname in model_files.items():
        if os.path.exists(fname):
            try:
                try:
                    models[name] = joblib.load(fname)
                except Exception:
                    with open(fname,"rb") as f:
                        models[name] = pickle.load(f)
                # success
            except Exception as e:
                warnings.append(f"Failed to load {fname}: {e}")
        else:
            warnings.append(f"Model file missing: {fname}")
    scaler = None
    if os.path.exists("scaler.pkl"):
        try:
            try:
                scaler = joblib.load("scaler.pkl")
            except Exception:
                with open("scaler.pkl","rb") as f:
                    scaler = pickle.load(f)
        except Exception as e:
            warnings.append(f"Failed to load scaler.pkl: {e}")
            scaler = None
    else:
        warnings.append("scaler.pkl not found.")
    return models, scaler, warnings

# ---------------- Load data & models ----------------
sch_df = load_scholarship_df()
reach_df = load_reach_df()
models, scaler, model_warnings = load_models_and_scaler()
for w in model_warnings:
    st.warning(w)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🏆 Eligibility Finder", "📊 Reach Predictor", "📈 Dashboard"])

# ---------------- TAB 1: Eligibility Finder ----------------
with tab1:
    st.header("🎯 Scholarship Eligibility Finder (Improved UI)")
    st.markdown("Enter student details below. Click **Find Eligible Scholarships** to view cards or table.")

    # Build dropdown options from data safely
    genders = ["All"] + sorted(set([g for g in sch_df['gender'].unique() if g and g.lower() != "nan"]))
    categories = ["All"] + sorted(set([c for c in sch_df['category'].unique() if c and c.lower() != "nan"]))
    edus = ["All"] + sorted(set([e for e in sch_df['education_level'].unique() if e and e.lower() != "nan"]))

    col1, col2, col3 = st.columns(3)
    with col1:
        input_gender = st.selectbox("Gender", genders, index=0)
        input_category = st.selectbox("Category", categories, index=0)
    with col2:
        input_income = st.number_input("Annual Family Income (₹)", min_value=0, value=150000, step=5000)
        input_edu = st.selectbox("Education Level", edus, index=0)
    with col3:
        search_term = st.text_input("Search scholarship name / description", "")

    def filter_eligible(df):
        df2 = df.copy()
        # income: use income_limit_numeric if present
        if "income_limit_numeric" in df2.columns:
            df2 = df2[(df2["income_limit_numeric"].isna()) | (input_income <= df2["income_limit_numeric"])]
        # gender
        if input_gender and input_gender != "All":
            df2 = df2[(df2["gender"].str.lower() == input_gender.lower()) | (df2["gender"].str.lower() == "all")]
        # category
        if input_category and input_category != "All":
            df2 = df2[(df2["category"].str.upper() == input_category.upper()) | (df2["category"].str.upper() == "ALL")]
        # education
        if input_edu and input_edu != "All":
            df2 = df2[df2["education_level"].str.contains(input_edu, case=False, na=False) | (df2["education_level"].str.lower() == "all")]
        # search
        if search_term:
            df2 = df2[df2["scholarship_name"].str.contains(search_term, case=False, na=False) | df2["description"].str.contains(search_term, case=False, na=False)]
        return df2

    if st.button("🔎 Find Eligible Scholarships"):
        eligible_df = filter_eligible(sch_df)
        st.session_state["eligible_df"] = eligible_df

        if eligible_df.empty:
            st.warning("No scholarships matched your criteria.")
        else:
            st.success(f"Found {len(eligible_df)} scholarships.")
            # counts
            tn_df = eligible_df[eligible_df["level"].str.contains("state|tn|tamil", case=False, na=False)]
            central_df = eligible_df[eligible_df["level"].str.contains("central|national", case=False, na=False)]
            c1, c2, c3 = st.columns(3)
            c1.metric("🏛️ Tamil Nadu", len(tn_df))
            c2.metric("🇮🇳 Central", len(central_df))
            c3.metric("📄 Total", len(eligible_df))

            st.markdown("---")
            view_mode = st.radio("View as:", ["Cards", "Table"], horizontal=True)

            if view_mode == "Cards":
                st.markdown("### 🟢 Tamil Nadu Scholarships")
                if not tn_df.empty:
                    for _, r in tn_df.iterrows():
                        web = r.get("website","").strip()
                        if web and not web.lower().startswith("http"):
                            web = "https://" + web
                        st.markdown(f"""
                        <div class="scholar-card tn">
                          <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="max-width:78%;">
                              <h4 style="margin:0">{r.get('scholarship_name','')}</h4>
                              <small>{r.get('provider','')}</small>
                              <p style="margin:6px 0 0 0;"><b>Category:</b> {r.get('category','')} &nbsp; | &nbsp; <b>Gender:</b> {r.get('gender','')}</p>
                              <p style="margin:6px 0 0 0;"><b>Edu:</b> {r.get('education_level','')} &nbsp; | &nbsp; <b>Income Limit:</b> {r.get('income_limit','')}</p>
                            </div>
                            <div style="text-align:right;">
                              <a class="apply-btn-tn" href="{web}" target="_blank">Apply</a>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No Tamil Nadu scholarships found.")

                st.markdown("### 🟣 Central Scholarships")
                if not central_df.empty:
                    for _, r in central_df.iterrows():
                        web = r.get("website","").strip()
                        if web and not web.lower().startswith("http"):
                            web = "https://" + web
                        st.markdown(f"""
                        <div class="scholar-card central">
                          <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="max-width:78%;">
                              <h4 style="margin:0">{r.get('scholarship_name','')}</h4>
                              <small>{r.get('provider','')}</small>
                              <p style="margin:6px 0 0 0;"><b>Category:</b> {r.get('category','')} &nbsp; | &nbsp; <b>Gender:</b> {r.get('gender','')}</p>
                              <p style="margin:6px 0 0 0;"><b>Edu:</b> {r.get('education_level','')} &nbsp; | &nbsp; <b>Income Limit:</b> {r.get('income_limit','')}</p>
                            </div>
                            <div style="text-align:right;">
                              <a class="apply-btn" href="{web}" target="_blank">Apply</a>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No Central scholarships found.")

            else:  # Table view — robust detection & display
                # create lowercased columns mapping for safe lookup
                df_tmp = eligible_df.copy()
                df_tmp.columns = df_tmp.columns.str.lower().str.strip()
                # possible keys and options
                col_map = {
                    "scholarship_name": ["scholarship_name","scholarship","name","title"],
                    "level": ["level","type","scheme"],
                    "category": ["category","caste","community"],
                    "gender": ["gender","sex"],
                    "education_level": ["education_level","education","eligible_classes"],
                    "income_limit": ["income_limit","income","income_limit_numeric"],
                    "amount": ["amount","scholarship_amount","amount_numeric"],
                    "website": ["website","link","url","application_link"],
                    "description": ["description","details","note"],
                    "provider": ["provider","agency","organisation","organization"]
                }
                matched = {}
                for key, opts in col_map.items():
                    for o in opts:
                        if o in df_tmp.columns:
                            matched[key] = o
                            break
                if not matched:
                    st.warning("No recognized columns found to display as table.")
                else:
                    display_cols = [v for k,v in matched.items()]
                    df_show = df_tmp[display_cols].rename(columns={v: k.replace("_"," ").title() for k,v in matched.items()})
                    st.markdown("### 📋 Eligible Scholarships — Table View")
                    st.dataframe(df_show.reset_index(drop=True), use_container_width=True)
                    # also provide markdown preview
                    try:
                        st.markdown(df_show.to_markdown(index=False), unsafe_allow_html=True)
                    except:
                        pass

# ---------------- TAB 2: Reach Predictor ----------------
with tab2:
    st.header("📊 Scholarship Reach Predictor (District-level)")
    if reach_df.empty:
        st.warning("Reach dataset not found (TN_Scholarship_Reach_REALISTIC.csv).")
    else:
        # ensure derived columns exist
        if 'income_to_infra' not in reach_df.columns and 'avg_family_income' in reach_df.columns and 'school_infrastructure_index' in reach_df.columns:
            reach_df['income_to_infra'] = reach_df['avg_family_income'] / reach_df['school_infrastructure_index'].replace(0,1)
        if 'awareness_index' not in reach_df.columns and 'literacy_rate' in reach_df.columns and 'schools_with_internet_percent' in reach_df.columns:
            reach_df['awareness_index'] = (reach_df['literacy_rate'] * reach_df['schools_with_internet_percent'])/100

        required = [
            "avg_family_income","literacy_rate","female_ratio","rural_population_percent",
            "num_students","schools_with_computer_lab_percent","schools_with_internet_percent",
            "school_infrastructure_index","income_to_infra","awareness_index"
        ]
        missing_feats = [f for f in required if f not in reach_df.columns]
        if missing_feats:
            st.warning(f"Reach dataset missing expected features: {missing_feats}. Predictions may be impacted.")

        if not models:
            st.error("No models loaded. Upload model .pkl files.")
        else:
            model_choice = st.selectbox("Choose Model", list(models.keys()))
            district = st.selectbox("Select District", reach_df["district"].unique())

            row = reach_df[reach_df["district"]==district].iloc[0]
            colA, colB = st.columns(2)
            with colA:
                avg_income = st.number_input("Average Family Income", value=float(row.get("avg_family_income",0)))
                literacy = st.number_input("Literacy Rate (%)", value=float(row.get("literacy_rate",0)))
                female_ratio = st.number_input("Female Ratio", value=float(row.get("female_ratio",0)))
                rural_pct = st.number_input("Rural Population (%)", value=float(row.get("rural_population_percent",0)))
                num_students = st.number_input("Number of Students", value=int(row.get("num_students",0)))
            with colB:
                comp_lab = st.number_input("Schools with Computer Lab (%)", value=float(row.get("schools_with_computer_lab_percent",0)))
                internet_pct = st.number_input("Schools with Internet (%)", value=float(row.get("schools_with_internet_percent",0)))
                infra_idx = st.number_input("School Infrastructure Index", value=float(row.get("school_infrastructure_index",0)))

            income_to_infra = avg_income / (infra_idx if infra_idx != 0 else 1)
            awareness_index = (literacy * internet_pct) / 100

            feat_vector = [avg_income, literacy, female_ratio, rural_pct, num_students, comp_lab, internet_pct, infra_idx, income_to_infra, awareness_index]
            X = np.array([feat_vector])

            # scale safely
            if scaler is not None:
                try:
                    Xs = scaler.transform(X)
                except Exception as e:
                    st.warning(f"Scaler transform failed: {e}. Using raw features.")
                    Xs = X
            else:
                Xs = X

            if st.button("🚀 Predict Reach"):
                model = models.get(model_choice)
                if model is None:
                    st.error("Selected model file not loaded.")
                else:
                    try:
                        pred = model.predict(Xs)[0]
                        st.success(f"🎯 Predicted Scholarship Reach in {district}: {pred:.2f}%")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        # optional: show correlation & performance if possible
        if scaler is not None and all(f in reach_df.columns for f in required) and "scholarship_reach_percent" in reach_df.columns:
            with st.expander("Model Performance & Correlation"):
                try:
                    X_all = scaler.transform(reach_df[required])
                    y_all = reach_df["scholarship_reach_percent"]
                    from sklearn.model_selection import train_test_split
                    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                    perf = []
                    for n,m in models.items():
                        try:
                            ypred = m.predict(Xte)
                            from sklearn.metrics import mean_squared_error, r2_score
                            rmse = np.sqrt(mean_squared_error(yte, ypred))
                            r2 = r2_score(yte, ypred)
                            perf.append({"Model":n,"RMSE":round(rmse,2),"R2":round(r2,2)})
                        except Exception:
                            perf.append({"Model":n,"RMSE":"NA","R2":"NA"})
                    st.table(pd.DataFrame(perf))
                except Exception as e:
                    st.warning(f"Could not compute performance: {e}")

# ---------------- TAB 3: Dashboard ----------------
with tab3:
    st.header("📈 Scholarship Data Dashboard")
    base_df = st.session_state.get("eligible_df", sch_df)

    st.subheader("Summary Metrics")
    colA, colB, colC = st.columns(3)
    colA.metric("Total Scholarships", len(base_df))
    colB.metric("Tamil Nadu (approx)", base_df['level'].str.contains("state|tn|tamil", case=False, na=False).sum())
    colC.metric("Central (approx)", base_df['level'].str.contains("central|national", case=False, na=False).sum())

    st.markdown("---")
    if PLOTLY:
        col_choice = st.selectbox("Column for bar chart", options=["category","gender","level","education_level"])
        vc = base_df[col_choice].value_counts().nlargest(20).reset_index()
        vc.columns = ["label","count"]
        fig = px.bar(vc, x="label", y="count", title=f"Top values in {col_choice}", labels={"label":col_choice,"count":"Count"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pie — Level distribution")
        lvl = base_df['level'].value_counts().reset_index()
        lvl.columns = ["level","count"]
        fig2 = px.pie(lvl, names="level", values="count", title="Level distribution")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Plotly not available — showing matplotlib charts.")
        vc = base_df['category'].value_counts().nlargest(20)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_title("Top categories")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Data (filtered / eligible)")
    st.dataframe(base_df.reset_index(drop=True), use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("**Developed by:** Logesh Kannan S  •  **Guide:** Dr. Rajkumar  •  Anna University Regional Campus, Madurai")
