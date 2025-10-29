# =========================================================
# 🎓 Anna University Scholarship App (Full Version)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 🏫 Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Anna University Scholarship App", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }
    .card {
        background-color: #eaf8f1;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# 🖼️ Header
# ---------------------------------------------------------
st.title("🎓 Anna University Scholarship App")
st.markdown("#### Eligibility Finder | Reach Predictor | Scholarship Dashboard")
st.markdown("---")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏆 Scholarship Eligibility Finder",
    "📊 Scholarship Reach Predictor",
    "📈 Scholarship Dashboard"
])

# =========================================================
# TAB 1 — Eligibility Finder
# =========================================================
with tab1:
    st.subheader("Find Scholarships You Are Eligible For")

    @st.cache_data
    def load_scholarship_data():
        df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
        df.columns = df.columns.str.strip()
        return df

    df_sch = load_scholarship_data()
    st.success(f"✅ {len(df_sch)} Scholarships Loaded Successfully!")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    category = st.selectbox("Community / Category", ["SC", "ST", "OBC", "General", "Minority"])
    income = st.number_input("Annual Family Income (₹)", min_value=0, value=200000)
    edu_level = st.selectbox("Education Level", ["School", "UG", "PG", "PhD"])

    if st.button("🔍 Find Eligible Scholarships"):
        def check_eligibility(row):
            try:
                limit = float(str(row["Income Limit"]).replace(",", "").strip())
            except:
                limit = 99999999
            if income > limit:
                return False
            if row["Gender"] != "All" and row["Gender"].lower() != gender.lower():
                return False
            if row["Category"] != "All" and row["Category"].upper() != category.upper():
                return False
            if row["Education Level"] != "All" and edu_level.lower() not in row["Education Level"].lower():
                return False
            return True

        eligible_df = df_sch[df_sch.apply(check_eligibility, axis=1)]

        if not eligible_df.empty:
            st.success(f"🎉 {len(eligible_df)} Scholarships Found Matching Your Profile")
            st.dataframe(eligible_df[["Scholarship Name", "Category", "Gender", "Education Level", "Amount", "Website"]])
            csv = eligible_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Eligible Scholarships", csv, "Eligible_Scholarships.csv", "text/csv")
            st.session_state["eligible_df"] = eligible_df
        else:
            st.warning("😔 No scholarships match your profile. Try adjusting your filters.")

# =========================================================
# TAB 2 — Scholarship Reach Predictor
# =========================================================
with tab2:
    st.subheader("Predict Scholarship Reach by District")

    @st.cache_data
    def load_reach_data():
        df = pd.read_csv("TN_Scholarship_Reach_REALISTIC.csv")
        df["income_to_infra"] = df["avg_family_income"] / df["school_infrastructure_index"].replace(0, 1)
        df["awareness_index"] = (df["literacy_rate"] * df["schools_with_internet_percent"]) / 100
        return df

    df_reach = load_reach_data()

    scaler = joblib.load("scaler.pkl")
    models = {
        "Linear Regression": joblib.load("Linear_Regression_model.pkl"),
        "Random Forest": joblib.load("Random_Forest_model.pkl"),
        "Gradient Boosting": joblib.load("Gradient_Boosting_model.pkl")
    }

    district = st.selectbox("Select District", df_reach["district"].unique())
    model_choice = st.selectbox("Select Model", list(models.keys()))

    dist_row = df_reach[df_reach["district"] == district].iloc[0]
    features = [
        "avg_family_income", "literacy_rate", "female_ratio", "rural_population_percent",
        "num_students", "schools_with_computer_lab_percent", "schools_with_internet_percent",
        "school_infrastructure_index"
    ]

    inputs = []
    for f in features:
        val = st.number_input(f"{f.replace('_', ' ').title()}", value=float(dist_row[f]))
        inputs.append(val)

    income_to_infra = inputs[0] / (inputs[-1] if inputs[-1] != 0 else 1)
    awareness_index = (inputs[1] * inputs[6]) / 100
    X = np.array([inputs + [income_to_infra, awareness_index]])
    X_scaled = scaler.transform(X)

    if st.button("🚀 Predict Scholarship Reach"):
        pred = models[model_choice].predict(X_scaled)[0]
        st.markdown(f"<div class='card'><h3>🎯 Predicted Scholarship Reach: {pred:.2f}%</h3></div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 — Dashboard (Advanced)
# =========================================================
with tab3:
    st.subheader("📈 Scholarship Dashboard & Visualization")

    df_dash = st.session_state.get("eligible_df", load_scholarship_data())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏛️ Tamil Nadu Scholarships", len(df_dash[df_dash["Level"].str.contains("State", case=False, na=False)]))
    with col2:
        st.metric("🇮🇳 Central Scholarships", len(df_dash[df_dash["Level"].str.contains("Central", case=False, na=False)]))
    with col3:
        st.metric("📊 Total Scholarships", len(df_dash))

    st.markdown("---")
    search = st.text_input("🔍 Search by Scholarship Name", "")
    cat = st.multiselect("🎯 Filter by Category", sorted(df_dash["Category"].dropna().unique()))
    gender = st.multiselect("🚻 Filter by Gender", sorted(df_dash["Gender"].dropna().unique()))
    edu = st.multiselect("🎓 Filter by Education Level", sorted(df_dash["Education Level"].dropna().unique()))

    filtered = df_dash.copy()
    if search:
        filtered = filtered[filtered["Scholarship Name"].str.contains(search, case=False, na=False)]
    if cat:
        filtered = filtered[filtered["Category"].isin(cat)]
    if gender:
        filtered = filtered[filtered["Gender"].isin(gender)]
    if edu:
        filtered = filtered[filtered["Education Level"].isin(edu)]

    st.write(f"### 🧾 {len(filtered)} Scholarships Found")

    # Tamil Nadu & Central split
    tn_df = filtered[filtered["Level"].str.contains("State", case=False, na=False)]
    central_df = filtered[filtered["Level"].str.contains("Central", case=False, na=False)]

    def show_cards(sub_df, title, bg):
        if not sub_df.empty:
            st.subheader(title)
            for _, r in sub_df.iterrows():
                st.markdown(
                    f"""
                    <div style='background-color:{bg};padding:15px;border-radius:12px;
                    margin-bottom:12px;box-shadow:2px 2px 10px rgba(0,0,0,0.1);'>
                      <h4 style='color:#0d6efd;'>{r["Scholarship Name"]}</h4>
                      <p><b>Category:</b> {r["Category"]} | <b>Gender:</b> {r["Gender"]}
                      | <b>Education:</b> {r["Education Level"]} <br>
                      <b>Income Limit:</b> ₹{r["Income Limit"]} | <b>Amount:</b> {r["Amount"]}</p>
                      <a href='{r["Website"]}' target='_blank'>
                        <button style='background-color:#4CAF50;color:white;padding:6px 14px;
                        border:none;border-radius:8px;cursor:pointer;'>🌐 Visit Website</button>
                      </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No scholarships in this section.")

    show_cards(tn_df, "🏛️ Tamil Nadu State Scholarships", "#f9f9f9")
    show_cards(central_df, "🇮🇳 Central Government Scholarships", "#f0f7ff")

    st.markdown("---")
    st.subheader("📊 Visual Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df_dash["Category"].value_counts())
    with col2:
        pie_data = df_dash["Gender"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **Developed by:** Logesh Kannan S  
    **Under Guidance:** Faculty, Anna University Regional Campus Madurai  
    **Purpose:** To enhance accessibility and awareness of scholarship opportunities.
    """
)









