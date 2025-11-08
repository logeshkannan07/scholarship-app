import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="Scholarship Recommendation & Reach Predictor",
    layout="wide"
)

# ----------------------------- PAGE STYLE -----------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #E6F0FA;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1, h2, h3, h4 {
    color: #003366;
}
.stButton>button {
    background-color: #004C99;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #0066CC;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------- LOAD DATA -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Curated_Scholarships_India_TN_200.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

@st.cache_resource
def load_model_and_scaler():
    with open("Linear_Regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

df = load_data()
model, scaler = load_model_and_scaler()

# ----------------------------- TITLE -----------------------------
st.title("🎓 Scholarship Recommendation & Reach Predictor")
st.image("anna-university-logo.png", width=120)
st.markdown("<h4 style='text-align:center;'>Empowering Students through AI-driven Scholarship Analysis</h4>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------- TABS -----------------------------
tab1, tab2 = st.tabs(["🏆 Eligibility & Reach Predictor", "📊 Data Visualization Dashboard"])

# ----------------------------- TAB 1 -----------------------------
with tab1:
    st.header("Scholarship Eligibility & Reach Prediction")
    st.write("Enter your details to check which scholarships you are eligible for and predict your reach score.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        education = st.selectbox("Education Level", sorted(df["education_level"].dropna().unique()))
    with col2:
        category = st.selectbox("Category", sorted(df["category"].dropna().unique()))
    with col3:
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))
    with col4:
        income = st.number_input("Annual Family Income (₹)", min_value=0, step=1000)

    # Eligibility filter
    eligible_df = df[
        (df["education_level"] == education)
        & (df["category"] == category)
        & ((df["gender"] == gender) | (df["gender"] == "All"))
        & (df["income_limit"] >= income)
    ]

    st.markdown("### 🎯 Eligible Scholarships")
    if not eligible_df.empty:
        tn_sch = eligible_df[eligible_df["scholarship_type"].str.contains("Tamil Nadu", case=False, na=False)]
        central_sch = eligible_df[eligible_df["scholarship_type"].str.contains("Central", case=False, na=False)]

        st.success(f"✅ Total Eligible Scholarships: {len(eligible_df)}")
        st.info(f"Tamil Nadu Scholarships: {len(tn_sch)} | Central Scholarships: {len(central_sch)}")

        st.dataframe(eligible_df[["scholarship_name", "scholarship_type", "education_level", "category", "income_limit"]])

        # Reach Prediction (keep same)
        st.subheader("📈 Reach Predictor")
        features = [[income]]
        scaled_features = scaler.transform(features)
        reach_pred = model.predict(scaled_features)[0]
        st.metric(label="Predicted Reach Score", value=f"{reach_pred:.2f}")

    else:
        st.warning("No scholarships found for the provided details.")

# ----------------------------- TAB 2 -----------------------------
with tab2:
    st.header("📊 Data Visualization Dashboard")
    st.write("Visualize the entire scholarship dataset using dynamic filters and interactive charts.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        type_filter = st.multiselect("Scholarship Type", sorted(df["scholarship_type"].dropna().unique()), default=df["scholarship_type"].unique())
    with col2:
        edu_filter = st.multiselect("Education Level", sorted(df["education_level"].dropna().unique()), default=df["education_level"].unique())
    with col3:
        cat_filter = st.multiselect("Category", sorted(df["category"].dropna().unique()), default=df["category"].unique())
    with col4:
        gender_filter = st.multiselect("Gender", sorted(df["gender"].dropna().unique()), default=df["gender"].unique())

    filtered_df = df[
        (df["scholarship_type"].isin(type_filter))
        & (df["education_level"].isin(edu_filter))
        & (df["category"].isin(cat_filter))
        & (df["gender"].isin(gender_filter))
    ]

    st.markdown(f"### Showing {len(filtered_df)} Scholarships after filters")

    col_a, col_b = st.columns(2)
    with col_a:
        type_chart = filtered_df["scholarship_type"].value_counts().reset_index()
        type_chart.columns = ["Scholarship Type", "Count"]
        fig1 = px.bar(type_chart, x="Scholarship Type", y="Count", color="Scholarship Type",
                      title="Scholarships by Type (Tamil Nadu vs Central)", text="Count")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Figure 1: Comparison of Tamil Nadu and Central scholarships available.")

    with col_b:
        edu_chart = filtered_df["education_level"].value_counts().reset_index()
        edu_chart.columns = ["Education Level", "Count"]
        fig2 = px.pie(edu_chart, values="Count", names="Education Level", title="Scholarships by Education Level")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Figure 2: Distribution of scholarships across education levels.")

    cat_chart = filtered_df["category"].value_counts().reset_index()
    cat_chart.columns = ["Category", "Count"]
    fig3 = px.bar(cat_chart, x="Category", y="Count", color="Category",
                  title="Scholarships by Category", text="Count")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Figure 3: Scholarships available for each category/community.")

    if "income_limit" in filtered_df.columns:
        fig4 = px.histogram(filtered_df, x="income_limit", nbins=10, title="Distribution of Income Limits")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Figure 4: Range of income limits defining scholarship eligibility.")

    if "gender" in filtered_df.columns:
        gender_chart = filtered_df["gender"].value_counts().reset_index()
        gender_chart.columns = ["Gender", "Count"]
        fig5 = px.pie(gender_chart, names="Gender", values="Count", title="Scholarships by Gender")
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Figure 5: Gender distribution of scholarships in the dataset.")

st.markdown("---")
st.markdown("🧠 *Developed as part of the AI-powered Scholarship Recommendation System Project (Anna University)*")
