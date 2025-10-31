import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# -----------------------------------------
# APP CONFIG
# -----------------------------------------
st.set_page_config(page_title="Scholarship Dashboard", layout="wide")
st.title("🎓 Anna University Scholarship Portal")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
    return df

df = load_data()

# Load Model for Reach Prediction
@st.cache_resource
def load_model():
    model = pickle.load(open("reach_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# -----------------------------------------
# TAB SETUP
# -----------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Scholarship Eligibility Finder",
    "📊 Scholarship Reach Prediction",
    "📈 Dataset Dashboard"
])

# -----------------------------------------
# TAB 1: SCHOLARSHIP ELIGIBILITY FINDER
# -----------------------------------------
with tab1:
    st.header("🎯 Find Your Eligible Scholarships")

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Select Your Category", sorted(df["Category"].dropna().unique()))
    with col2:
        gender = st.selectbox("Select Your Gender", sorted(df["Gender"].dropna().unique()))
    with col3:
        income = st.number_input("Enter Annual Income (₹)", min_value=0, step=1000)

    edu = st.selectbox("Select Education Level", sorted(df["Education Level"].dropna().unique()))

    # Filter data
    eligible_df = df[
        (df["Category"].str.contains(category, case=False, na=False)) &
        (df["Gender"].str.contains(gender, case=False, na=False)) &
        (df["Education Level"].str.contains(edu, case=False, na=False)) &
        (df["Income Limit"] >= income)
    ]

    if eligible_df.empty:
        st.warning("No scholarships found matching your criteria.")
    else:
        st.success(f"✅ Found {len(eligible_df)} scholarships for you!")

        # Separate TN and Central
        tn_df = eligible_df[eligible_df["Level"].str.contains("State", case=False, na=False)]
        central_df = eligible_df[eligible_df["Level"].str.contains("Central", case=False, na=False)]

        # --- Display Tamil Nadu Scholarships ---
        if not tn_df.empty:
            st.subheader("🟢 Tamil Nadu Scholarships")
            for _, row in tn_df.iterrows():
                st.markdown(f"""
                <div style="background-color:#e7f7e7;padding:15px;border-radius:10px;margin-bottom:10px;">
                    <h4>🎓 {row['Scholarship Name']}</h4>
                    <p><b>Amount:</b> {row['Amount']}</p>
                    <p><b>Category:</b> {row['Category']} | <b>Gender:</b> {row['Gender']}</p>
                    <p><b>Income Limit:</b> ₹{row['Income Limit']} | <b>Education Level:</b> {row['Education Level']}</p>
                    <a href="{row['Website']}" target="_blank"><b>🔗 Apply Now</b></a>
                </div>
                """, unsafe_allow_html=True)

        # --- Display Central Scholarships ---
        if not central_df.empty:
            st.subheader("🟣 Central Scholarships")
            for _, row in central_df.iterrows():
                st.markdown(f"""
                <div style="background-color:#efe7f7;padding:15px;border-radius:10px;margin-bottom:10px;">
                    <h4>🏛️ {row['Scholarship Name']}</h4>
                    <p><b>Amount:</b> {row['Amount']}</p>
                    <p><b>Category:</b> {row['Category']} | <b>Gender:</b> {row['Gender']}</p>
                    <p><b>Income Limit:</b> ₹{row['Income Limit']} | <b>Education Level:</b> {row['Education Level']}</p>
                    <a href="{row['Website']}" target="_blank"><b>🔗 Apply Now</b></a>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------
# TAB 2: REACH PREDICTION
# -----------------------------------------
with tab2:
    st.header("📊 Scholarship Reach Prediction")

    district = st.selectbox("Select Your District", ["Chennai", "Madurai", "Coimbatore", "Salem", "Trichy"])
    st.write("Predicting reach rate for selected district...")

    sample_features = np.array([[len(district), 2025]])  # Dummy example for demo
    scaled = scaler.transform(sample_features)
    reach_pred = model.predict(scaled)[0]

    st.metric(label="Predicted Reach Rate (%)", value=f"{reach_pred:.2f}%")

# -----------------------------------------
# TAB 3: DATASET DASHBOARD
# -----------------------------------------
with tab3:
    st.header("📈 Dataset Dashboard and Visualization")

    st.write("### Explore the Full Scholarship Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("🔹 Scholarships by Level")
    fig1 = px.pie(df, names="Level", title="Distribution of Scholarships by Level")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🔹 Scholarships by Education Level")
    fig2 = px.bar(df, x="Education Level", color="Level", title="Scholarships Count by Education Level")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔹 Income Limit Distribution")
    fig3 = px.box(df, y="Income Limit", color="Level", title="Income Limit Distribution by Scholarship Level")
    st.plotly_chart(fig3, use_container_width=True)

    st.info("Use the graphs to analyze how scholarships vary by level, education, and income limit.")

# -----------------------------------------
# END OF APP
# -----------------------------------------
st.caption("Developed by Logesh Kannan | Guided by Dr. Rajkumar | Anna University © 2025")


