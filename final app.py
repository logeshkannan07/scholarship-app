import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="Scholarship Dashboard", layout="wide")

# =====================================================
# LOAD MODELS AND SCALER
# =====================================================
@st.cache_resource
def load_all_models():
    models = {}
    try:
        models["Linear Regression"] = pickle.load(open("Linear_Regression_model.pkl", "rb"))
    except:
        st.warning("Linear Regression model not found.")

    try:
        models["Gradient Boosting"] = pickle.load(open("Gradient_Boosting_model.pkl", "rb"))
    except:
        st.warning("Gradient Boosting model not found.")

    try:
        scaler = pickle.load(open("scaler.pkl", "rb"))
    except:
        scaler = None
        st.warning("Scaler file not found.")
    return models, scaler


models, scaler = load_all_models()

# =====================================================
# LOAD DATASETS
# =====================================================
@st.cache_data
def load_data():
    df_elig = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
    df_reach = pd.read_csv("TN_Scholarship_Reach_REALISTIC.csv")
    return df_elig, df_reach


df_elig, df_reach = load_data()

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs([
    "🎓 Scholarship Eligibility Checker",
    "📈 Scholarship Reach Prediction",
    "📊 Dataset Visualization Dashboard"
])

# =====================================================
# TAB 1 — SCHOLARSHIP ELIGIBILITY CHECKER
# =====================================================
with tab1:
    st.header("🎯 Find Your Eligible Scholarships")

    # User Input Section
    gender = st.selectbox("Select Gender", ["Male", "Female", "All"])
    category = st.selectbox("Select Category", sorted(df_elig["Category"].dropna().unique()))
    income = st.number_input("Enter Annual Family Income (₹)", min_value=0, value=200000)
    education = st.selectbox("Select Education Level", sorted(df_elig["Education Level"].dropna().unique()))

    # Filter Data
    eligible = df_elig[
        ((df_elig["Gender"] == gender) | (df_elig["Gender"] == "All")) &
        ((df_elig["Category"] == category) | (df_elig["Category"] == "All")) &
        (df_elig["Income Limit"] >= income) &
        (df_elig["Education Level"].str.contains(education, case=False, na=False))
    ]

    # Display Sectioned by Level
    st.subheader("🎓 Tamil Nadu Level Scholarships")
    tn_sch = eligible[eligible["Level"].str.contains("State", case=False, na=False)]
    if not tn_sch.empty:
        for _, row in tn_sch.iterrows():
            st.markdown(
                f"""
                <div style="background-color:#e7f3ff;padding:15px;margin-bottom:10px;border-radius:10px;">
                    <b>{row['Scholarship Name']}</b><br>
                    <i>{row['Description']}</i><br>
                    <b>Amount:</b> ₹{row['Amount']} | 
                    <b>Category:</b> {row['Category']} | 
                    <b>Income Limit:</b> ₹{row['Income Limit']}<br>
                    <a href="https://{row['Website']}" target="_blank" style="color:#0073e6;">🔗 Apply Now</a>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No Tamil Nadu scholarships found for your inputs.")

    st.subheader("🏛️ Central Level Scholarships")
    central_sch = eligible[eligible["Level"].str.contains("Central|National", case=False, na=False)]
    if not central_sch.empty:
        for _, row in central_sch.iterrows():
            st.markdown(
                f"""
                <div style="background-color:#fff5e6;padding:15px;margin-bottom:10px;border-radius:10px;">
                    <b>{row['Scholarship Name']}</b><br>
                    <i>{row['Description']}</i><br>
                    <b>Amount:</b> ₹{row['Amount']} | 
                    <b>Category:</b> {row['Category']} | 
                    <b>Income Limit:</b> ₹{row['Income Limit']}<br>
                    <a href="https://{row['Website']}" target="_blank" style="color:#ff6600;">🔗 Apply Now</a>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No Central scholarships found for your inputs.")


# =====================================================
# TAB 2 — SCHOLARSHIP REACH PREDICTION
# =====================================================
with tab2:
    st.header("📈 Predict Scholarship Reach (District-wise)")

    feature_cols = [
        "avg_family_income", "literacy_rate", "female_ratio",
        "rural_population_percent", "num_students",
        "schools_with_computer_lab_percent", "schools_with_internet_percent",
        "school_infrastructure_index", "income_to_infra", "awareness_index"
    ]

    # Derived columns
    df_reach["income_to_infra"] = df_reach["avg_family_income"] / df_reach["school_infrastructure_index"].replace(0, 1)
    df_reach["awareness_index"] = (df_reach["literacy_rate"] * df_reach["schools_with_internet_percent"]) / 100

    district = st.selectbox("Select District", df_reach["district"].unique())
    model_choice = st.selectbox("Choose Model", list(models.keys()))

    dist_data = df_reach[df_reach["district"] == district].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        avg_income = st.number_input("Average Family Income", value=float(dist_data["avg_family_income"]))
        literacy = st.slider("Literacy Rate (%)", 0.0, 100.0, float(dist_data["literacy_rate"]))
        female = st.slider("Female Ratio", 800.0, 1100.0, float(dist_data["female_ratio"]))
        rural = st.slider("Rural Population (%)", 0.0, 100.0, float(dist_data["rural_population_percent"]))
        students = st.number_input("Number of Students", value=int(dist_data["num_students"]))
    with col2:
        comp = st.slider("Schools with Computer Lab (%)", 0.0, 100.0, float(dist_data["schools_with_computer_lab_percent"]))
        net = st.slider("Schools with Internet (%)", 0.0, 100.0, float(dist_data["schools_with_internet_percent"]))
        infra = st.slider("Infrastructure Index", 0.0, 100.0, float(dist_data["school_infrastructure_index"]))

        income_to_infra = avg_income / (infra if infra != 0 else 1)
        awareness = (literacy * net) / 100

    features = np.array([[avg_income, literacy, female, rural, students, comp, net, infra, income_to_infra, awareness]])

    if scaler:
        try:
            features = scaler.transform(features)
        except:
            st.warning("Scaler transformation failed — using raw features.")

    if st.button("🚀 Predict Reach"):
        model = models.get(model_choice)
        if model:
            try:
                pred = model.predict(features)[0]
                st.success(f"🎯 Predicted Scholarship Reach in {district}: **{pred:.2f}%**")
            except Exception as e:
                st.error(f"Prediction Error: {e}")


# =====================================================
# TAB 3 — VISUALIZATION DASHBOARD
# =====================================================
with tab3:
    st.header("📊 Scholarship Dataset Dashboard")
    st.markdown("Explore the overall dataset insights with visualizations.")

    chart_type = st.selectbox("Choose Visualization Type", ["Bar Chart", "Pie Chart", "Line Chart"])
    column_choice = st.selectbox("Select Column for Analysis", df_elig.columns)

    if chart_type == "Bar Chart":
        fig = px.bar(df_elig[column_choice].value_counts().reset_index(),
                     x="index", y=column_choice, title=f"Bar Chart of {column_choice}")
    elif chart_type == "Pie Chart":
        fig = px.pie(df_elig[column_choice].value_counts().reset_index(),
                     names="index", values=column_choice, title=f"Pie Chart of {column_choice}")
    else:
        fig = px.line(df_elig[column_choice].value_counts().reset_index(),
                      x="index", y=column_choice, title=f"Line Chart of {column_choice}")

    st.plotly_chart(fig, use_container_width=True)




