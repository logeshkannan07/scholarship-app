# =========================================================
# 🎓 Combined Scholarship App | Anna University
# =========================================================

import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 🏫 Page Config & Logo
# ---------------------------------------------------------
st.image("anna_logo.png", width=150)
st.set_page_config(page_title="Scholarship App", page_icon="🎓", layout="centered")
st.image("https://upload.wikimedia.org/wikipedia/en/7/7a/Anna_University_Logo.png", width=120)
st.title("🎓 Anna University Scholarship App")

# ---------------------------------------------------------
# 🔖 Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🏆 Scholarship Eligibility Finder", "📊 Scholarship Reach Predictor"])

# =========================================================
# TAB 1: Scholarship Eligibility Finder
# =========================================================
with tab1:
    st.markdown("#### Anna University Regional Campus, Madurai")
    
    @st.cache_data
    def load_scholarship_data():
        df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
        df.columns = df.columns.str.strip()
        return df

    df_scholarship = load_scholarship_data()
    st.success(f"✅ Loaded {len(df_scholarship)} scholarships successfully!")

    # Normalization maps
    gender_map = {"male": "Male", "female": "Female", "other": "Other"}
    category_map = {"sc": "SC", "st": "ST", "obc": "OBC", "gen": "General", "minority": "Minority"}
    edu_map = {"school": "School", "ug": "UG", "pg": "PG", "phd": "PhD"}

    st.subheader("🧾 Enter Your Details")
    col1, col2 = st.columns(2)
    with col1:
        gender_in = st.selectbox("Gender", ["Male", "Female", "Other"])
        income = st.number_input("Annual Family Income (₹)", min_value=0, value=150000)
    with col2:
        category_in = st.selectbox("Community / Category", ["SC", "ST", "OBC", "General", "Minority"])
        education_in = st.selectbox("Education Level", ["School", "UG", "PG", "PhD"])

    if st.button("🔍 Find Eligible Scholarships"):
        gender = gender_map.get(gender_in.lower(), "All")
        category = category_map.get(category_in.lower(), "All")
        education = edu_map.get(education_in.lower(), "All")

        def eligible(row):
            if row["Gender"] != "All" and row["Gender"].lower() != gender.lower():
                return False
            if row["Category"] != "All" and row["Category"].upper() != category.upper():
                return False
            try:
                limit = float(str(row["Income_Limit"]).replace(",", "").strip())
                if income > limit:
                    return False
            except:
                pass
            if row["Eligible_Classes"] != "All" and education.lower() not in row["Eligible_Classes"].lower():
                return False
            return True

        result_df = df_scholarship[df_scholarship.apply(eligible, axis=1)]

        if not result_df.empty:
            st.success(f"🎉 Found {len(result_df)} scholarships matching your profile!")
            st.dataframe(result_df[["Scholarship_Name", "Provider", "Amount", "Application_Link", "Last_Date"]], use_container_width=True)
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Eligible Scholarships as CSV", data=csv, file_name="Eligible_Scholarships.csv", mime="text/csv")
        else:
            st.warning("😔 No scholarships matched your details. Try adjusting inputs.")

# =========================================================
# TAB 2: Scholarship Reach Predictor
# =========================================================
with tab2:
    st.subheader("Predict Scholarship Reach in TN Districts")

    @st.cache_data
    def load_reach_data():
        df = pd.read_csv("TN_Scholarship_Reach_REALISTIC.csv")
        df["income_to_infra"] = df["avg_family_income"] / df["school_infrastructure_index"].replace(0,1)
        df["awareness_index"] = (df["literacy_rate"] * df["schools_with_internet_percent"]) / 100
        return df

    df_reach = load_reach_data()
    
    feature_cols = ["avg_family_income","literacy_rate","female_ratio","rural_population_percent",
                    "num_students","schools_with_computer_lab_percent","schools_with_internet_percent",
                    "school_infrastructure_index","income_to_infra","awareness_index"]
    X = df_reach[feature_cols]
    y = df_reach["scholarship_reach_percent"]

    scaler = joblib.load("scaler.pkl") if st.button("Load Existing Scaler") else StandardScaler().fit(X)
    
    # Train or load models
    model_files = ["Linear_Regression_model.pkl", "Random_Forest_model.pkl", "Gradient_Boosting_model.pkl"]
    trained_models = {}
    for file in model_files:
        try:
            trained_models[file.replace("_model.pkl","").replace("_"," ")] = joblib.load(file)
        except:
            st.warning(f"Model file {file} not found. Train models first.")

    # Dummy areas
    dummy_areas = {
        "Chennai": ["Adyar", "T. Nagar", "Velachery"],
        "Madurai": ["Simmakkal", "Alanganallur", "Thiruparankundram"],
        "Coimbatore": ["RS Puram", "Peelamedu", "Gandhipuram"]
    }

    model_choice = st.selectbox("Choose Model", list(trained_models.keys()))
    district = st.selectbox("Select District", df_reach["district"].unique())
    area = st.selectbox("Select Area / City", dummy_areas.get(district, ["Area 1", "Area 2"]))
    st.write(f"Selected Area: **{area}, {district}**")

    district_data = df_reach[df_reach["district"]==district].iloc[0]

    avg_income = st.number_input("Average Family Income", value=float(district_data["avg_family_income"]))
    literacy_rate = st.slider("Literacy Rate (%)", 0.0, 100.0, float(district_data["literacy_rate"]))
    female_ratio = st.slider("Female Ratio", 800.0, 1100.0, float(district_data["female_ratio"]))
    rural_percent = st.slider("Rural Population (%)", 0.0, 100.0, float(district_data["rural_population_percent"]))
    num_students = st.number_input("Number of Students", value=int(district_data["num_students"]))
    comp_lab_percent = st.slider("Schools with Computer Lab (%)", 0.0, 100.0, float(district_data["schools_with_computer_lab_percent"]))
    internet_percent = st.slider("Schools with Internet (%)", 0.0, 100.0, float(district_data["schools_with_internet_percent"]))
    infra_index = st.slider("School Infrastructure Index", 0.0, 100.0, float(district_data["school_infrastructure_index"]))

    income_to_infra = avg_income / (infra_index if infra_index!=0 else 1)
    awareness_index = (literacy_rate * internet_percent)/100

    if st.button("Predict Scholarship Reach"):
        features_array = np.array([[avg_income, literacy_rate, female_ratio, rural_percent,
                                    num_students, comp_lab_percent, internet_percent, infra_index,
                                    income_to_infra, awareness_index]])
        features_scaled = scaler.transform(features_array)
        model = trained_models[model_choice]
        pred = model.predict(features_scaled)[0]
        pred = float(np.clip(pred,0,100))
        st.success(f"🏆 Predicted Scholarship Reach: {pred:.2f}%")

    if st.checkbox("Show Correlation Heatmap"):
        corr = df_reach[feature_cols + ["scholarship_reach_percent"]].corr()
        fig, ax = plt.subplots(figsize=(9,7))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Model Performance"):
        res=[]
        X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        for name, m in trained_models.items():
            y_pred = m.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test,y_pred))
            r2 = r2_score(y_test,y_pred)
            res.append({"Model":name,"RMSE":round(rmse,2),"R²":round(r2,2)})
        st.table(pd.DataFrame(res))

