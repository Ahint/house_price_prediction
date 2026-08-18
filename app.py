import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for matplotlib
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["font.family"] = "sans-serif"

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Real Estate Valuation AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Modern minimalist CSS
st.markdown(
    """
<style>
    /* Root colors - vibrant modern palette */
    :root {
        --primary: #1E40AF;
        --secondary: #7C3AED;
        --accent: #F59E0B;
        --success: #10B981;
        --danger: #EF4444;
    }
    
    /* Global styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 64, 175, 0.2);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border-color: #D1D5DB;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 2px solid #F3F4F6;
        padding-bottom: 1rem;
    }
    
    /* Result card - prominent */
    .result-card {
        background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        border: none;
        text-align: center;
        box-shadow: 0 16px 40px rgba(30, 64, 175, 0.3);
        margin: 2rem 0;
    }
    
    .result-label {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.85;
    }
    
    .result-price {
        font-size: 4rem;
        font-weight: 900;
        margin: 1rem 0;
        letter-spacing: -2px;
    }
    
    .result-subtext {
        font-size: 1rem;
        opacity: 0.8;
    }
    
    /* Grid layout for inputs */
    .input-grid {
        display: grid;
        gap: 1rem;
    }
    
    /* Slider container */
    .slider-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .slider-hint {
        font-size: 0.8rem;
        color: #9CA3AF;
        margin-top: 0.3rem;
        font-style: italic;
    }
    
    /* Stats section */
    .stat-box {
        background: #F9FAFB;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary);
        margin-bottom: 1rem;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .stat-value {
        font-size: 1.5rem;
        color: #1F2937;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(30, 64, 175, 0.3) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #E5E7EB, transparent);
        margin: 2rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F9FAFB !important;
        border-radius: 8px !important;
    }
    
    /* Feature category badges */
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .badge-location {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    
    .badge-building {
        background-color: #FED7AA;
        color: #92400E;
    }
    
    .badge-environment {
        background-color: #D1FAE5;
        color: #065F46;
    }
    
    .badge-economic {
        background-color: #FCE7F3;
        color: #831843;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


model, scaler = load_model_and_scaler()

# ============================================================================
# FEATURE INFO
# ============================================================================
features_data = {
    "CRIM": {
        "name": "Crime Rate",
        "unit": "per capita",
        "min": 0,
        "max": 30,
        "step": 0.1,
        "category": "Location",
    },
    "ZN": {
        "name": "Residential Zoning",
        "unit": "% of land",
        "min": 0,
        "max": 100,
        "step": 1,
        "category": "Location",
    },
    "INDUS": {
        "name": "Non-Retail Business",
        "unit": "% acres",
        "min": 0,
        "max": 30,
        "step": 0.1,
        "category": "Building",
    },
    "CHAS": {
        "name": "Charles River Proximity",
        "unit": "binary",
        "min": 0,
        "max": 1,
        "step": 1,
        "category": "Location",
    },
    "NOX": {
        "name": "Air Pollution (NOX)",
        "unit": "ppm/10",
        "min": 0.3,
        "max": 0.9,
        "step": 0.01,
        "category": "Environment",
    },
    "RM": {
        "name": "Rooms per Dwelling",
        "unit": "count",
        "min": 3,
        "max": 10,
        "step": 0.1,
        "category": "Building",
    },
    "AGE": {
        "name": "Building Age",
        "unit": "% pre-1940",
        "min": 0,
        "max": 100,
        "step": 1,
        "category": "Building",
    },
    "DIS": {
        "name": "Distance to Jobs",
        "unit": "weighted miles",
        "min": 0,
        "max": 15,
        "step": 0.1,
        "category": "Location",
    },
    "RAD": {
        "name": "Highway Access",
        "unit": "index (1-24)",
        "min": 1,
        "max": 24,
        "step": 1,
        "category": "Location",
    },
    "TAX": {
        "name": "Property Tax",
        "unit": "per $10k",
        "min": 150,
        "max": 710,
        "step": 10,
        "category": "Economic",
    },
    "PTRATIO": {
        "name": "Student-Teacher Ratio",
        "unit": "ratio",
        "min": 12,
        "max": 22,
        "step": 0.1,
        "category": "Economic",
    },
    "B": {
        "name": "Demographics Index",
        "unit": "index",
        "min": 0,
        "max": 400,
        "step": 1,
        "category": "Environment",
    },
    "LSTAT": {
        "name": "Low Income Population",
        "unit": "% residents",
        "min": 0,
        "max": 40,
        "step": 0.1,
        "category": "Economic",
    },
}

# Initialize session state
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "CRIM": 5.0,
        "ZN": 20.0,
        "INDUS": 10.0,
        "CHAS": 0,
        "NOX": 0.5,
        "RM": 6.5,
        "AGE": 40.0,
        "DIS": 5.0,
        "RAD": 8,
        "TAX": 300,
        "PTRATIO": 16.0,
        "B": 350.0,
        "LSTAT": 12.0,
    }

# ============================================================================
# HEADER
# ============================================================================
st.markdown(
    """
<div class="header-container">
    <h1 class="header-title">🏡 Real Estate Valuation AI</h1>
    <p class="header-subtitle">Intelligent property price prediction powered by machine learning</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# MAIN CONTENT
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🎯 Valuation", "📊 Analytics", "ℹ️ Guide"])

feature_order = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]

with tab1:
    # Prediction first (prominent)
    col_left, col_right = st.columns([1, 1.2], gap="large")

    # ========== LEFT COLUMN: QUICK INPUTS ==========
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">⚡ Quick Settings</div>', unsafe_allow_html=True
        )

        # Key features in cards
        col_a, col_b = st.columns(2, gap="small")

        with col_a:
            st.markdown(
                """
            <div class="stat-box">
                <div class="stat-label">Rooms</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.session_state.inputs["RM"] = st.slider(
                "Rooms per Dwelling",
                min_value=3.0,
                max_value=10.0,
                value=st.session_state.inputs["RM"],
                step=0.1,
                label_visibility="collapsed",
            )

        with col_b:
            st.markdown(
                """
            <div class="stat-box">
                <div class="stat-label">Crime</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.session_state.inputs["CRIM"] = st.slider(
                "Crime Rate",
                min_value=0.0,
                max_value=30.0,
                value=st.session_state.inputs["CRIM"],
                step=0.1,
                label_visibility="collapsed",
            )

        col_c, col_d = st.columns(2, gap="small")

        with col_c:
            st.markdown(
                """
            <div class="stat-box">
                <div class="stat-label">Building Age</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.session_state.inputs["AGE"] = st.slider(
                "Building Age",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.inputs["AGE"],
                step=1.0,
                label_visibility="collapsed",
            )

        with col_d:
            st.markdown(
                """
            <div class="stat-box">
                <div class="stat-label">Low Income %</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.session_state.inputs["LSTAT"] = st.slider(
                "Lower Status Population",
                min_value=0.0,
                max_value=40.0,
                value=st.session_state.inputs["LSTAT"],
                step=0.1,
                label_visibility="collapsed",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.markdown('<div class="card">', unsafe_allow_html=True)

            cols = st.columns(2, gap="small")
            with cols[0]:
                st.session_state.inputs["INDUS"] = st.slider(
                    "Non-Retail Business %",
                    min_value=0.0,
                    max_value=30.0,
                    value=st.session_state.inputs["INDUS"],
                    step=0.1,
                )
                st.session_state.inputs["DIS"] = st.slider(
                    "Distance to Jobs",
                    min_value=0.0,
                    max_value=15.0,
                    value=st.session_state.inputs["DIS"],
                    step=0.1,
                )
                st.session_state.inputs["NOX"] = st.slider(
                    "Air Pollution (NOX)",
                    min_value=0.3,
                    max_value=0.9,
                    value=st.session_state.inputs["NOX"],
                    step=0.01,
                )

            with cols[1]:
                st.session_state.inputs["ZN"] = st.slider(
                    "Residential Zoning %",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.inputs["ZN"],
                    step=1.0,
                )
                st.session_state.inputs["RAD"] = st.slider(
                    "Highway Access",
                    min_value=1,
                    max_value=24,
                    value=st.session_state.inputs["RAD"],
                    step=1,
                )
                st.session_state.inputs["TAX"] = st.slider(
                    "Property Tax",
                    min_value=150,
                    max_value=710,
                    value=st.session_state.inputs["TAX"],
                    step=10,
                )

            st.session_state.inputs["PTRATIO"] = st.slider(
                "Student-Teacher Ratio",
                min_value=12.0,
                max_value=22.0,
                value=st.session_state.inputs["PTRATIO"],
                step=0.1,
            )

            st.session_state.inputs["B"] = st.slider(
                "Demographics Index",
                min_value=0.0,
                max_value=400.0,
                value=st.session_state.inputs["B"],
                step=1.0,
            )

            st.session_state.inputs["CHAS"] = st.selectbox(
                "River Proximity",
                options=[0, 1],
                format_func=lambda x: "Adjacent" if x == 1 else "Not Adjacent",
                index=int(st.session_state.inputs["CHAS"]),
            )

            st.markdown("</div>", unsafe_allow_html=True)

    # ========== RIGHT COLUMN: PREDICTION RESULT ==========
    with col_right:
        # Prepare features
        features = np.array([[st.session_state.inputs[f] for f in feature_order]])

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        # Display prominent result
        st.markdown(
            f"""
        <div class="result-card">
            <div class="result-label">💰 Estimated Property Value</div>
            <div class="result-price">${prediction:,.0f}</div>
            <div class="result-subtext">Based on property characteristics</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Key stats
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">📈 Key Metrics</div>', unsafe_allow_html=True
        )

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric(label="Estimated Value", value=f"${prediction:,.0f}", delta=None)

        with col_stat2:
            avg_price = 22.5 * 10000  # Rough average
            diff = ((prediction - avg_price) / avg_price) * 100
            st.metric(
                label="vs. Average",
                value=f"{diff:+.1f}%",
                delta=f"${prediction - avg_price:+,.0f}",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Action buttons
        col_b1, col_b2 = st.columns(2, gap="small")
        with col_b1:
            if st.button("📋 View Summary", use_container_width=True):
                st.session_state.show_summary = True

        with col_b2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.inputs = {
                    "CRIM": 5.0,
                    "ZN": 20.0,
                    "INDUS": 10.0,
                    "CHAS": 0,
                    "NOX": 0.5,
                    "RM": 6.5,
                    "AGE": 40.0,
                    "DIS": 5.0,
                    "RAD": 8,
                    "TAX": 300,
                    "PTRATIO": 16.0,
                    "B": 350.0,
                    "LSTAT": 12.0,
                }
                st.rerun()

    # Show summary if requested
    if st.session_state.get("show_summary", False):
        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">📋 Input Summary & Visualization</div>',
            unsafe_allow_html=True,
        )

        summary_data = []
        for feature in feature_order:
            info = features_data[feature]
            value = st.session_state.inputs[feature]
            # Calculate percentage within range
            pct = ((value - info["min"]) / (info["max"] - info["min"])) * 100
            summary_data.append(
                {
                    "Feature": features_data[feature]["name"],
                    "Value": f"{value:.2f}",
                    "Unit": features_data[feature]["unit"],
                    "Category": features_data[feature]["category"],
                    "Percentage": pct,
                }
            )

        df_summary = pd.DataFrame(summary_data)

        # Create radar-like chart using matplotlib
        categories = [features_data[f]["name"] for f in feature_order]
        percentages = [
            (
                (st.session_state.inputs[f] - features_data[f]["min"])
                / (features_data[f]["max"] - features_data[f]["min"])
            )
            * 100
            for f in feature_order
        ]

        # Create subplot: table and bar chart
        col_t1, col_t2 = st.columns([1.2, 1])

        with col_t1:
            st.subheader("📊 Input Values")
            st.dataframe(
                df_summary[["Feature", "Value", "Unit"]],
                use_container_width=True,
                hide_index=True,
            )

        with col_t2:
            # Horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, 10))
            colors = ["#7C3AED" if x >= 50 else "#F59E0B" for x in percentages]
            ax.barh(
                categories, percentages, color=colors, edgecolor="white", linewidth=1.5
            )
            ax.set_xlabel("Percentage in Range (%)", fontsize=11, fontweight="600")
            ax.set_title("Property Profile", fontsize=13, fontweight="700", pad=20)
            ax.set_xlim(0, 100)
            ax.grid(axis="x", alpha=0.3, linestyle="--")

            # Add percentage labels
            for i, v in enumerate(percentages):
                ax.text(
                    v + 2, i, f"{v:.0f}%", va="center", fontsize=9, fontweight="600"
                )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ========== ANALYTICS TAB ==========
with tab2:
    # Feature Sensitivity Analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">📈 Feature Sensitivity Analysis</div>',
        unsafe_allow_html=True,
    )

    sensitivity_feature = st.selectbox(
        "Select feature to analyze:",
        options=["RM", "CRIM", "LSTAT", "NOX", "AGE", "DIS", "RAD", "TAX", "PTRATIO"],
        format_func=lambda x: features_data[x]["name"],
    )

    # Generate sensitivity data
    sensitivity_data = []
    feature_info = features_data[sensitivity_feature]

    # Create range of values for the selected feature
    value_range = np.linspace(feature_info["min"], feature_info["max"], 30)

    for val in value_range:
        temp_inputs = st.session_state.inputs.copy()
        temp_inputs[sensitivity_feature] = val

        features_temp = np.array([[temp_inputs[f] for f in feature_order]])
        features_temp_scaled = scaler.transform(features_temp)
        pred_temp = model.predict(features_temp_scaled)[0]

        sensitivity_data.append({"value": val, "price": pred_temp})

    df_sensitivity = pd.DataFrame(sensitivity_data)

    # Create sensitivity chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df_sensitivity["value"],
        df_sensitivity["price"],
        color="#7C3AED",
        linewidth=3,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#7C3AED",
        markeredgewidth=2,
    )
    ax.fill_between(
        df_sensitivity["value"], df_sensitivity["price"], alpha=0.2, color="#7C3AED"
    )

    ax.set_xlabel(
        f"{feature_info['name']} ({feature_info['unit']})",
        fontsize=12,
        fontweight="600",
    )
    ax.set_ylabel("Estimated Price ($)", fontsize=12, fontweight="600")
    ax.set_title(
        f"Price Sensitivity to {feature_info['name']}",
        fontsize=14,
        fontweight="700",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x / 1e3:.0f}K"))

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Compare Scenarios
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">🔍 Scenario Comparison</div>', unsafe_allow_html=True
    )

    col_s1, col_s2, col_s3 = st.columns(3, gap="small")

    with col_s1:
        scenario1_rm = st.slider("Scenario 1: Rooms", 3, 10, 5, key="s1_rm")
        scenario1_crime = st.slider("Scenario 1: Crime", 0, 30, 5, key="s1_crime")
    with col_s2:
        scenario2_rm = st.slider("Scenario 2: Rooms", 3, 10, 7, key="s2_rm")
        scenario2_crime = st.slider("Scenario 2: Crime", 0, 30, 3, key="s2_crime")
    with col_s3:
        scenario3_rm = st.slider("Scenario 3: Rooms", 3, 10, 8, key="s3_rm")
        scenario3_crime = st.slider("Scenario 3: Crime", 0, 30, 1, key="s3_crime")

    # Calculate predictions for scenarios
    scenarios_data = []
    scenario_prices = []

    for i, (rooms, crime) in enumerate(
        [
            (scenario1_rm, scenario1_crime),
            (scenario2_rm, scenario2_crime),
            (scenario3_rm, scenario3_crime),
        ],
        1,
    ):
        temp_inputs = st.session_state.inputs.copy()
        temp_inputs["RM"] = rooms
        temp_inputs["CRIM"] = crime

        features_temp = np.array([[temp_inputs[f] for f in feature_order]])
        features_temp_scaled = scaler.transform(features_temp)
        pred_temp = model.predict(features_temp_scaled)[0]
        scenario_prices.append(pred_temp)

        scenarios_data.append(
            {
                "Scenario": f"Scenario {i}",
                "Rooms": f"{rooms:.1f}",
                "Crime Rate": f"{crime:.1f}",
                "Price": pred_temp,
            }
        )

    # Display table
    df_scenarios = pd.DataFrame(scenarios_data)
    col_table1, col_table2 = st.columns([2, 1])

    with col_table1:
        st.dataframe(
            df_scenarios[["Scenario", "Rooms", "Crime Rate"]],
            use_container_width=True,
            hide_index=True,
        )

    with col_table2:
        st.markdown("**Estimated Prices:**")
        for i, price in enumerate(scenario_prices, 1):
            st.metric(f"Scenario {i}", f"${price:,.0f}")

    # Scenario comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = [f"Scenario {i}" for i in range(1, 4)]
    colors_scenarios = ["#1E40AF", "#7C3AED", "#F59E0B"]
    bars = ax.bar(
        scenarios,
        scenario_prices,
        color=colors_scenarios,
        edgecolor="white",
        linewidth=2,
    )

    # Add value labels on bars
    for bar, price in zip(bars, scenario_prices):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${price:,.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="600",
        )

    ax.set_ylabel("Estimated Price ($)", fontsize=12, fontweight="600")
    ax.set_title("Scenario Price Comparison", fontsize=14, fontweight="700", pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x / 1e3:.0f}K"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">🎯 Feature Impact Analysis</div>',
        unsafe_allow_html=True,
    )

    # Calculate impact of each feature
    impacts = []
    feature_names = []

    base_inputs = st.session_state.inputs.copy()
    features_base = np.array([[base_inputs[f] for f in feature_order]])
    features_base_scaled = scaler.transform(features_base)
    base_price = model.predict(features_base_scaled)[0]

    for feature in feature_order:
        info = features_data[feature]
        # Test with max value
        test_inputs = base_inputs.copy()
        test_inputs[feature] = info["max"]

        features_test = np.array([[test_inputs[f] for f in feature_order]])
        features_test_scaled = scaler.transform(features_test)
        test_price = model.predict(features_test_scaled)[0]

        impact = ((test_price - base_price) / base_price) * 100
        impacts.append(impact)
        feature_names.append(features_data[feature]["name"])

    # Create impact dataframe
    df_impacts = pd.DataFrame(
        {"Feature": feature_names, "Impact %": impacts}
    ).sort_values("Impact %", key=abs, ascending=True)

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(9, 8))
    colors_impact = ["#10B981" if x > 0 else "#EF4444" for x in df_impacts["Impact %"]]
    bars = ax.barh(
        df_impacts["Feature"],
        df_impacts["Impact %"],
        color=colors_impact,
        edgecolor="white",
        linewidth=1.5,
    )

    ax.set_xlabel("Price Impact (%)", fontsize=12, fontweight="600")
    ax.set_title(
        "Feature Impact on Price\n(% change when feature at maximum)",
        fontsize=13,
        fontweight="700",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Add value labels
    for i, (feature, impact) in enumerate(
        zip(df_impacts["Feature"], df_impacts["Impact %"])
    ):
        ax.text(
            impact + (1 if impact > 0 else -1),
            i,
            f"{impact:.1f}%",
            va="center",
            ha="left" if impact > 0 else "right",
            fontsize=9,
            fontweight="600",
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Price Distribution Analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">💰 Price Range Distribution</div>',
        unsafe_allow_html=True,
    )

    # Generate random property variations
    np.random.seed(42)
    prices_distribution = []

    for _ in range(500):
        variation = st.session_state.inputs.copy()
        # Add random variations
        for feature in feature_order:
            info = features_data[feature]
            variation[feature] += np.random.normal(
                0, (info["max"] - info["min"]) * 0.15
            )
            variation[feature] = np.clip(variation[feature], info["min"], info["max"])

        features_var = np.array([[variation[f] for f in feature_order]])
        features_var_scaled = scaler.transform(features_var)
        price_var = model.predict(features_var_scaled)[0]
        prices_distribution.append(price_var)

    # Current price
    current_price = prediction

    # Create histogram with current price line
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        prices_distribution,
        bins=30,
        color="#7C3AED",
        alpha=0.7,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.axvline(
        current_price,
        color="#F59E0B",
        linestyle="--",
        linewidth=3,
        label="Your Property",
    )

    ax.set_xlabel("Estimated Price ($)", fontsize=12, fontweight="600")
    ax.set_ylabel("Number of Properties", fontsize=12, fontweight="600")
    ax.set_title(
        "Price Distribution Analysis - Similar Properties",
        fontsize=14,
        fontweight="700",
        pad=20,
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x / 1e3:.0f}K"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Statistics
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.metric("Mean Price", f"${np.mean(prices_distribution):,.0f}")
    with col_stat2:
        st.metric("Median Price", f"${np.median(prices_distribution):,.0f}")
    with col_stat3:
        st.metric("Min Price", f"${np.min(prices_distribution):,.0f}")
    with col_stat4:
        st.metric("Max Price", f"${np.max(prices_distribution):,.0f}")

    st.markdown("</div>", unsafe_allow_html=True)

# ========== INFO TAB ==========
with tab3:
    col_info1, col_info2 = st.columns([1, 1], gap="large")

    with col_info1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">🎓 About This Tool</div>', unsafe_allow_html=True
        )
        st.markdown("""
This AI-powered valuation tool predicts real estate prices based on property characteristics using machine learning.
        
**What it does:**
- Analyzes 13 key property features
- Applies a trained ML model for predictions
- Provides instant price estimates
- Helps compare different scenarios
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">⚠️ Disclaimer</div>', unsafe_allow_html=True
        )
        st.markdown("""
This tool provides **estimates only**. Actual property values depend on:
- Market conditions
- Specific location details
- Property condition & renovations
- Recent comparable sales
- Market trends

Always verify with professional appraisals!
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">📚 Feature Descriptions</div>', unsafe_allow_html=True
    )

    features_to_show = [
        "CRIM",
        "ZN",
        "INDUS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]

    for feature in features_to_show:
        info = features_data[feature]
        st.markdown(f"**{info['name']} ({feature})**")
        st.caption(f"Unit: {info['unit']} | Range: {info['min']} - {info['max']}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
