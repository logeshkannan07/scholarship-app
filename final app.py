# =========================================================
# 🎓 Anna University Scholarship App (Final Enhanced Version)
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 🏫 Page Config & Styles
# ---------------------------------------------------------
st.set_page_config(page_title="Scholarship App", page_icon="🎓", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .card {
        background-color: #d1e7dd;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .scholar-card {
        background-color:#f8f9fa;
        border-left:6px solid #0d6efd;
        padding:15px;margin:10px 0;border-radius:8px;
    }
    .scholar-card h4 {margin:0;}
    .tn {border-left-color:#198754;}
    .central {border-left-color:#dc3545;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🖼️ Header
# ---------------------------------------------------------
st.image("anna-university-logo.png", width=120)
st.title("🎓 Anna University - Scholarship App")
st.markdown("#### Eligibility Finder | Reach Predictor | Data Dashboard")
st.markdown("---")

# ---------------------------------------------------------
# 🔖 Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏆 Scholarship Eligibility Finder",
    "📊 Scholarship Reach Predictor",
    "📈 Data Dashboard"
])

# =========================================================
# TAB 1: Scholarship Eligibility Finder
# =========================================================
with tab1:
    @st.cache_data
    def load_scholarship_data():
        df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    df_elig = load_scholarship_data()
    st.success(f"✅ Loaded {len(df_elig)} scholarships successfully!")

    st.subheader("🧾 Enter Your Details")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male","Female","Other"])
        income = st.number_input("Annual Family Income (₹)", min_value=0, value=150000)
    with col2:
        category = st.selectbox("Community / Category", ["SC","ST","OBC","General","Minority"])
        education = st.selectbox("Education Level", sorted(df_elig["education_level"].dropna().unique()))

    if st.button("🔍 Find Eligible Scholarships"):
        def eligible(row):
            if row.get("gender","All") not in ["All", gender]:
                return False
            if row.get("category","All") not in ["All", category]:
                return False
            try:
                limit = float(str(row.get("income_limit","0")).replace(",","").strip())
                if income > limit:
                    return False
            except:
                pass
            if row.get("education_level","All") not in ["All", education]:
                return False
            return True

        eligible_df = df_elig[df_elig.apply(eligible, axis=1)]
        if eligible_df.empty:
            st.warning("😔 No scholarships matched your details.")
        else:
            st.success(f"🎉 Found {len(eligible_df)} eligible scholarships!")
            view_mode = st.radio("View as:", ["Dashboard", "Table"], horizontal=True)

            if view_mode == "Dashboard":
                tn_df = eligible_df[eligible_df["level"].str.contains("State", case=False, na=False)]
                central_df = eligible_df[eligible_df["level"].str.contains("Central", case=False, na=False)]

                if not tn_df.empty:
                    st.markdown("### 🟢 Tamil Nadu Scholarships")
                    for _, r in tn_df.iterrows():
                        st.markdown(
                            f"""
                            <div class="scholar-card tn">
                            <h4>{r['scholarship_name']}</h4>
                            <b>Category:</b> {r['category']} | <b>Gender:</b> {r['gender']}<br>
                            <b>Income Limit:</b> ₹{r['income_limit']} | <b>Amount:</b> ₹{r['amount']}<br>
                            🌐 <a href="{r['website']}" target="_blank">Apply Now</a>
                            </div>
                            """, unsafe_allow_html=True)

                if not central_df.empty:
                    st.markdown("### 🔵 Central Scholarships")
                    for _, r in central_df.iterrows():
                        st.markdown(
                            f"""
                            <div class="scholar-card central">
                            <h4>{r['scholarship_name']}</h4>
                            <b>Category:</b> {r['category']} | <b>Gender:</b> {r['gender']}<br>
                            <b>Income Limit:</b> ₹{r['income_limit']} | <b>Amount:</b> ₹{r['amount']}<br>
                            🌐 <a href="{r['website']}" target="_blank">Apply Now</a>
                            </div>
                            """, unsafe_allow_html=True)

            else:  # --- Table View Fix ---
                desired_cols = [
                    "scholarship_name","level","category","gender",
                    "education_level","income_limit","amount","website"
                ]
                show_cols = [c for c in desired_cols if c in eligible_df.columns]
                if not show_cols:
                    st.warning("No suitable columns found to display as a table.")
                else:
                    rename_map = {
                        "scholarship_name":"Scholarship Name",
                        "education_level":"Education Level",
                        "income_limit":"Income Limit",
                        "amount":"Amount (₹)",
                        "website":"Website"
                    }
                    df_to_show = eligible_df[show_cols].rename(columns=rename_map)
                    st.dataframe(df_to_show.reset_index(drop=True), use_container_width=True)
                    st.markdown(df_to_show.to_markdown(index=False), unsafe_allow_html=True)

# =========================================================
# TAB 2: Scholarship Reach Predictor
# =========================================================
with tab2:
    st.subheader("Predict Scholarship Reach Across TN Districts")

    @st.cache_data
    def load_reach_data():
        df = pd.read_csv("TN_Scholarship_Reach_REALISTIC.csv")
        df["income_to_infra"] = df["avg_family_income"] / df["school_infrastructure_index"].replace(0,1)
        df["awareness_index"] = (df["literacy_rate"] * df["schools_with_internet_percent"]) / 100
        return df

    df_reach = load_reach_data()
    feature_cols = [
        "avg_family_income","literacy_rate","female_ratio","rural_population_percent",
        "num_students","schools_with_computer_lab_percent","schools_with_internet_percent",
        "school_infrastructure_index","income_to_infra","awareness_index"
    ]

    # load scaler & models
    scaler = joblib.load("scaler.pkl")
    models = {
        "Linear Regression": joblib.load("Linear_Regression_model.pkl"),
        "Gradient Boosting": joblib.load("Gradient_Boosting_model.pkl"),
        "Random Forest": joblib.load("Random_Forest_model.pkl")
    }

    model_choice = st.selectbox("Choose Model", list(models.keys()))
    district = st.selectbox("Select District", df_reach["district"].unique())

    dist = df_reach[df_reach["district"]==district].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        avg_income = st.number_input("Avg Family Income", value=float(dist["avg_family_income"]))
        literacy = st.slider("Literacy Rate (%)", 0.0, 100.0, float(dist["literacy_rate"]))
        female_ratio = st.slider("Female Ratio", 800.0, 1100.0, float(dist["female_ratio"]))
        rural = st.slider("Rural Population (%)", 0.0, 100.0, float(dist["rural_population_percent"]))
    with col2:
        students = st.number_input("No. of Students", value=int(dist["num_students"]))
        lab = st.slider("Schools with Computer Lab (%)", 0.0, 100.0, float(dist["schools_with_computer_lab_percent"]))
        net = st.slider("Schools with Internet (%)", 0.0, 100.0, float(dist["schools_with_internet_percent"]))
        infra = st.slider("Infrastructure Index", 0.0, 100.0, float(dist["school_infrastructure_index"]))

    income_to_infra = avg_income / (infra if infra != 0 else 1)
    awareness_index = (literacy * net) / 100
    features = np.array([[avg_income,literacy,female_ratio,rural,students,lab,net,infra,income_to_infra,awareness_index]])
    scaled = scaler.transform(features)
    model = models[model_choice]
    pred = float(np.clip(model.predict(scaled)[0],0,100))

    st.markdown(f"""
    <div class="card">
    <h3>🏆 Predicted Scholarship Reach in {district}: {pred:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Model Performance"):
        res=[]
        X = scaler.transform(df_reach[feature_cols])
        y = df_reach["scholarship_reach_percent"]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
        for n,m in models.items():
            ypred=m.predict(Xte)
            rmse=np.sqrt(mean_squared_error(yte,ypred))
            r2=r2_score(yte,ypred)
            res.append({"Model":n,"RMSE":round(rmse,2),"R²":round(r2,2)})
        st.table(pd.DataFrame(res))

# =========================================================
# TAB 3: Dashboard
# =========================================================
with tab3:
    st.header("📈 Scholarship Data Dashboard")
    df_dash = load_scholarship_data()
    st.dataframe(df_dash, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        col_choice = st.selectbox("Select Column for Bar Chart", ["category","gender","level","education_level"])
        st.bar_chart(df_dash[col_choice].value_counts())
    with col2:
        st.subheader("🥧 Pie Chart")
        pie_data = df_dash[col_choice].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
        st.pyplot(fig)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
**Developed by:** Logesh Kannan S  
**Guided by:** Faculty – Anna University Regional Campus Madurai  
**Purpose:** Enhancing Scholarship Accessibility & Awareness
""")







