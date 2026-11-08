
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EV Range Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - PREMIUM LIGHT THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

[data-testid="stMainBlockContainer"] {
    max-width: 100%;
    padding: 2rem 3rem;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    min-width: 250px;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 22px;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    min-height: 120px;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 15px !important;
}

[data-testid="stMetricValue"] {
    color: #059669 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

h1 {
    color: #123047 !important;
    font-size: 34px !important;
}

h2, h3 {
    color: #183b56 !important;
}

.stButton > button {
    background-color: #059669;
    color: white;
    border-radius: 9px;
    border: none;
    min-height: 45px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #047857;
    color: white;
    border: none;
}

hr {
    border-color: #e2e8f0;
}

[data-testid="stDataFrame"] {
    width: 100%;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("models/ev_range_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("dataset/india_ev_range_dataset.csv")


try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Unable to load model or dataset: {e}")
    st.stop()


TARGET = "real_world_range_km"

FEATURES = [
    col for col in df.columns
    if col not in [
        TARGET,
        "energy_consumption_kWh_per_km"
    ]
]

CATEGORICAL = [
    col for col in [
        "make",
        "model",
        "variant",
        "city_road_type",
        "terrain_type",
        "traffic_condition",
        "driving_style",
        "ac_usage"
    ]
    if col in FEATURES
]

NUMERICAL = [
    col for col in FEATURES
    if col not in CATEGORICAL
]


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("EV Intelligence")

st.sidebar.caption("Electric Vehicle Analytics Platform")

st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Range Prediction",
        "Data Analytics",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "AI-powered real-world EV range estimation"
)


# ==========================================
# HEADER
# ==========================================

st.title("⚡ EV Range Intelligence")

st.markdown(
    "Smart Electric Vehicle Range Prediction & Analytics Platform"
)

st.divider()


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.subheader("Dashboard Overview")

    total_records = len(df)
    total_models = df["model"].nunique()
    total_manufacturers = df["make"].nunique()
    average_range = df[TARGET].mean()

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4, gap="large")

    c1.metric("Total Records", f"{total_records:,}")
    c2.metric("EV Models", total_models)
    c3.metric("Manufacturers", total_manufacturers)
    c4.metric("Average Range", f"{average_range:.1f} km")

    st.divider()

    # RANGE DISTRIBUTION

    st.subheader("EV Range Distribution")

    fig = px.histogram(
        df,
        x=TARGET,
        nbins=30,
        title="Real-World EV Range Distribution",
        color_discrete_sequence=["#10b981"]
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=60, b=30),
        xaxis_title="Range (km)",
        yaxis_title="Number of Records"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # TWO COLUMN CHARTS

    left, right = st.columns(2, gap="large")

    with left:

        st.subheader("Manufacturer Comparison")

        manufacturer_data = (
            df.groupby("make")[TARGET]
            .mean()
            .reset_index()
            .sort_values(TARGET, ascending=False)
        )

        fig = px.bar(
            manufacturer_data,
            x="make",
            y=TARGET,
            color=TARGET,
            color_continuous_scale="Greens",
            title="Average Range by Manufacturer"
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title="Manufacturer",
            yaxis_title="Average Range (km)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("ARAI vs Real-World Range")

        fig = px.scatter(
            df,
            x="ARAI_certified_range_km",
            y=TARGET,
            color="make",
            hover_data=["model"],
            title="Certified Range vs Actual Estimated Range"
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title="ARAI Range (km)",
            yaxis_title="Real-World Range (km)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("EV Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# RANGE PREDICTION PAGE
# ==========================================

elif page == "Range Prediction":

    st.subheader("EV Range Prediction")

    st.write(
        "Enter your vehicle specifications and driving conditions."
    )

    user_input = {}

    left, right = st.columns(2, gap="large")

    with left:

        st.markdown("###Vehicle Information")

        for feature in ["make", "model", "variant"]:

            if feature in CATEGORICAL:

                options = sorted(
                    df[feature].dropna().unique().tolist(),
                    key=str
                )

                user_input[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options
                )

        for feature in [
            "battery_capacity_kWh",
            "battery_capacity_kWh_new",
            "motor_power_kW",
            "vehicle_weight_kg",
            "price_lakh_INR"
        ]:

            if feature in NUMERICAL:

                user_input[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(df[feature].median())
                )

    with right:

        st.markdown("### Driving Conditions")

        for feature in [
            "city_road_type",
            "terrain_type",
            "traffic_condition",
            "driving_style",
            "ac_usage"
        ]:

            if feature in CATEGORICAL:

                options = sorted(
                    df[feature].dropna().unique().tolist(),
                    key=str
                )

                user_input[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options,
                    format_func=lambda x: str(x).replace("_", " ").title()
                )

        for feature in [
            "drag_coefficient_Cd",
            "frontal_area_m2",
            "rolling_resistance_Cr",
            "motor_efficiency_pct",
            "ARAI_certified_range_km",
            "avg_driving_speed_kmh",
            "ambient_temperature_C",
            "passengers",
            "battery_age_years",
            "tyre_pressure_deviation_psi"
        ]:

            if feature in NUMERICAL:

                user_input[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(df[feature].median())
                )

    st.divider()

    if st.button("🔍 Predict EV Range", use_container_width=True):

        try:

            input_df = pd.DataFrame([user_input])
            input_df = input_df[FEATURES]

            prediction = float(model.predict(input_df)[0])

            st.success("Prediction completed successfully!")

            result1, result2 = st.columns(2)

            result1.metric(
                "Predicted Real-World Range",
                f"{prediction:.2f} km"
            )

            arai_value = user_input.get(
                "ARAI_certified_range_km", 0
            )

            difference = prediction - arai_value

            result2.metric(
                "Difference from ARAI Range",
                f"{difference:.2f} km"
            )

            # GAUGE CHART

            gauge = px.pie(
                values=[
                    max(prediction, 0),
                    max(df[TARGET].max() - prediction, 0)
                ],
                names=["Predicted Range", "Remaining"],
                hole=0.75,
                color_discrete_sequence=[
                    "#10b981",
                    "#e5e7eb"
                ]
            )

            gauge.update_layout(
                template="plotly_white",
                title="Predicted Range Indicator",
                height=300,
                showlegend=True
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            st.info(
                "This is an ML-based estimate. Actual driving range "
                "may vary with road, weather and battery conditions."
            )

            # DOWNLOAD PREDICTION REPORT

            report = pd.DataFrame([{
                "Manufacturer": user_input.get("make"),
                "Model": user_input.get("model"),
                "Variant": user_input.get("variant"),
                "Predicted Range (km)": round(prediction, 2),
                "ARAI Range (km)": arai_value
            }])

            st.download_button(
                "⬇️ Download Prediction Report",
                data=report.to_csv(index=False),
                file_name="EV_Prediction_Report.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ==========================================
# DATA ANALYTICS PAGE
# ==========================================

elif page == "Data Analytics":

    st.subheader("EV Data Analytics")

    st.write(
        "Explore relationships between EV specifications and range."
    )

    left, right = st.columns(2, gap="large")

    with left:

        st.markdown("### Battery Capacity vs Range")

        fig = px.scatter(
            df,
            x="battery_capacity_kWh",
            y=TARGET,
            color="make",
            title="Battery Capacity vs Real-World Range",
            hover_data=["model"]
        )

        fig.update_layout(
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("### 🚦 Driving Style Analysis")

        style_data = (
            df.groupby("driving_style")[TARGET]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            style_data,
            x="driving_style",
            y=TARGET,
            color=TARGET,
            title="Average Range by Driving Style",
            color_continuous_scale="Greens"
        )

        fig.update_layout(
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Feature Correlation Heatmap")

    numeric_df = df.select_dtypes(include="number")

    fig = px.imshow(
        numeric_df.corr(),
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        title="Numerical Feature Correlation",
        aspect="auto"
    )

    fig.update_layout(
        template="plotly_white",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("📋 Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download EV Dataset",
        data=df.to_csv(index=False),
        file_name="EV_Dataset.csv",
        mime="text/csv"
    )
    # Feature Importance Analysis

st.subheader("Top Factors Affecting EV Range")

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance", ascending=False
    ).head(10)

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Important Features",
        color="Importance",
        color_continuous_scale="Greens"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### EV Battery Range Prediction

    This project uses Machine Learning to estimate the
    real-world driving range of Electric Vehicles.

    **Technologies Used**

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Random Forest Regression
    - Streamlit
    - Plotly

    **Dataset Information**

    - 7,291 records
    - 45 EV models
    - 18 manufacturers
    - 25 original dataset columns

    **Model**

    Random Forest Regressor

    **Important Note**

    The dataset includes modelled driving scenarios.
    Predictions are estimates and should not be treated
    as guaranteed real-world vehicle performance.
    """)

    st.divider()

    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "Random Forest")
    c2.metric("R² Score", "0.9563")
    c3.metric("MAE", "19.90 km")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center> EV Range Intelligence | Machine Learning Project</center>",
    unsafe_allow_html=True
)
