# app.py
"""Master Enterprise Credit Risk Intelligence Platform Streamlit Cockpit."""
import time
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Setup page config
st.set_page_config(
    page_title="Credit Risk Intelligence Platform | NeoStats",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.helpers import format_currency, format_percent, fico_to_risk_tier, prob_to_credit_score
from src.utils.docker_utils import get_system_health
from src.data.loader import DataLoader, get_db_connection
from src.data.preprocessor import DataPreprocessor
from src.ml.predict import RiskPredictor
from src.ml.explain import ModelExplainer
from src.ml.rules import RuleEngine
from src.talk_to_data.nl_to_sql import NLToSQLAgent
from src.ml.train import train_all_models

logger = get_logger("StreamlitApp")

# ==============================================================================
# CUSTOM ENTERPRISE FINTECH CSS (DARK SLATE PALETTE, MODERN GLASSMORPHISM)
# ==============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Base */
    .stApp {
        background-color: #080C14;
        color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.025em;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Top Navbar Header */
    .platform-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .platform-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .platform-subtitle {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Metrics / KPI Cards */
    .kpi-card {
        background: linear-gradient(180deg, #131B2E 0%, #0F172A 100%);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #38BDF8;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Decision Badges */
    .badge-auto-approve {
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .badge-manual-review {
        background: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .badge-decline {
        background: rgba(239, 68, 68, 0.12);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .badge-latency {
        background: rgba(14, 165, 233, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(14, 165, 233, 0.35);
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0E1524;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid rgba(51, 65, 85, 0.5);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94A3B8;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background-color: rgba(255, 255, 255, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E293B 0%, #172033 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }

    /* Underwriter Summary Card */
    .narrative-card {
        background: linear-gradient(180deg, #131B2E 0%, #0F172A 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Preset Selection Buttons */
    .preset-chip {
        display: inline-block;
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 10px;
        margin-right: 6px;
        font-size: 0.8rem;
        color: #94A3B8;
    }

    /* Telemetry Sidebar Card */
    .sidebar-telemetry {
        background: #0E1524;
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# CACHED RESOURCE INITIALIZATION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_platform():
    """Initializes DuckDB, trained models, explainers, and NL-to-SQL engine."""
    Config.ensure_directories()
    loader = DataLoader()
    loader.ingest_to_duckdb()
    
    # Train models if artifacts are missing
    if not Config.CHAMPION_MODEL_PATH.exists() or not Config.PREPROCESSOR_PATH.exists():
        logger.info("Artifacts missing on startup. Triggering automated model training...")
        train_all_models()
        
    predictor = RiskPredictor()
    explainer = ModelExplainer()
    rule_engine = RuleEngine()
    nl_agent = NLToSQLAgent()
    
    return loader, predictor, explainer, rule_engine, nl_agent

# Load platform components
with st.spinner("Initializing DuckDB In-Memory OLAP & Machine Learning Engine..."):
    loader, predictor, explainer, rule_engine, nl_agent = initialize_platform()

# ==============================================================================
# TOP HEADER COCKPIT
# ==============================================================================
st.markdown("""
<div class="platform-header">
    <div>
        <div class="platform-title">CREDIT RISK INTELLIGENCE PLATFORM</div>
        <div class="platform-subtitle">Enterprise Quantitative Risk Modeling, In-Memory OLAP & Agentic SQL Copilot</div>
    </div>
    <div style="text-align: right;">
        <span class="badge-auto-approve">[STATUS: OPERATIONAL]</span>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 5px; font-family: 'JetBrains Mono', monospace;">
            DuckDB OLAP &bull; LightGBM GBDT &bull; Groq LLaMA 3.3
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# MASTER NAVIGATION TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "PORTFOLIO INTELLIGENCE & EDA",
    "REAL-TIME APPLICANT SCORER & XAI",
    "CREDIT POLICY RULE ENGINE",
    "TALK-TO-DATA AI COPILOT"
])

# ==============================================================================
# TAB 1: PORTFOLIO INTELLIGENCE & 5 KEY BANKING INSIGHTS
# ==============================================================================
with tab1:
    st.markdown("### Executive Portfolio Overview")
    
    # Top Level KPI Metrics
    try:
        summary = loader.get_portfolio_summary()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Portfolio Applications</div>
                <div class="kpi-value">{summary['total_loans']:,}</div>
                <div class="kpi-sub">Active Columnar Records</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Portfolio Default Rate</div>
                <div class="kpi-value" style="color: #F87171;">{summary['overall_default_rate_pct']}%</div>
                <div class="kpi-sub">{summary['total_defaulters']:,} Empirical Defaulters</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Credit Exposure</div>
                <div class="kpi-value">{format_currency(summary['total_portfolio_exposure'] / 1e9, symbol='$')}B</div>
                <div class="kpi-sub">Avg Principal: {format_currency(summary['avg_credit_amount'])}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg Applicant Annual Income</div>
                <div class="kpi-value">{format_currency(summary['avg_applicant_income'])}</div>
                <div class="kpi-sub">Avg Annuity: {format_currency(summary['avg_monthly_annuity'])}</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading summary metrics: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### The 5 Key Quantitative Risk Insights")
    
    # Insight 1 & 2
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("#### 1. Default Rate by Income Quintile")
        df_inc = loader.execute_query("""
            SELECT 
                CASE 
                    WHEN AMT_INCOME_TOTAL < 112500 THEN 'Q1 (<$112.5k)'
                    WHEN AMT_INCOME_TOTAL BETWEEN 112500 AND 147150 THEN 'Q2 ($112.5k-$147k)'
                    WHEN AMT_INCOME_TOTAL BETWEEN 147150 AND 180000 THEN 'Q3 ($147k-$180k)'
                    WHEN AMT_INCOME_TOTAL BETWEEN 180000 AND 225000 THEN 'Q4 ($180k-$225k)'
                    ELSE 'Q5 (>$225k)'
                END AS income_bracket,
                COUNT(*) AS applicants,
                ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct
            FROM loan_applications
            GROUP BY income_bracket
            ORDER BY income_bracket;
        """)
        fig_inc = px.bar(
            df_inc, x="income_bracket", y="default_rate_pct",
            text="default_rate_pct",
            color="default_rate_pct",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            labels={"income_bracket": "Income Quintile", "default_rate_pct": "Default Rate (%)"}
        )
        fig_inc.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(19, 27, 46, 0.8)",
            paper_bgcolor="rgba(19, 27, 46, 0.8)",
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        fig_inc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig_inc, use_container_width=True)
        st.caption("Lower income brackets demonstrate 1.8x higher default frequency relative to top earners.")

    with row1_col2:
        st.markdown("#### 2. Composite External Bureau Rating (EXT_SOURCES_MEAN)")
        df_ext = loader.execute_query("""
            SELECT 
                ROUND(COALESCE((COALESCE(EXT_SOURCE_1, 0.5) + COALESCE(EXT_SOURCE_2, 0.5) + COALESCE(EXT_SOURCE_3, 0.5))/3.0, 0.5), 1) AS bureau_rating_bucket,
                COUNT(*) AS applicants,
                ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct
            FROM loan_applications
            GROUP BY bureau_rating_bucket
            ORDER BY bureau_rating_bucket ASC;
        """)
        fig_ext = px.line(
            df_ext, x="bureau_rating_bucket", y="default_rate_pct",
            markers=True,
            line_shape="spline",
            labels={"bureau_rating_bucket": "Composite Bureau Rating (0.0 to 1.0)", "default_rate_pct": "Default Rate (%)"}
        )
        fig_ext.update_traces(line_color="#00E5FF", line_width=3, marker=dict(size=8, color="#38BDF8"))
        fig_ext.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(19, 27, 46, 0.8)",
            paper_bgcolor="rgba(19, 27, 46, 0.8)",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_ext, use_container_width=True)
        st.caption("Strong negative monotonic relationship: Default risk drops from >25% at rating 0.1 down to <2% above 0.8.")

    # Insight 3 & 4
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown("#### 3. Default Rate Across Age Cohorts")
        df_age = loader.execute_query("""
            SELECT 
                CASE 
                    WHEN ABS(DAYS_BIRTH)/365.25 < 30 THEN '1. Under 30'
                    WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 30 AND 39.99 THEN '2. 30s (30-39)'
                    WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 40 AND 49.99 THEN '3. 40s (40-49)'
                    WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 50 AND 59.99 THEN '4. 50s (50-59)'
                    ELSE '5. 60 and Above'
                END AS age_cohort,
                COUNT(*) AS applicants,
                ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct
            FROM loan_applications
            GROUP BY age_cohort
            ORDER BY age_cohort;
        """)
        fig_age = px.bar(
            df_age, x="age_cohort", y="default_rate_pct",
            text="default_rate_pct",
            color="default_rate_pct",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            labels={"age_cohort": "Age Cohort", "default_rate_pct": "Default Rate (%)"}
        )
        fig_age.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(19, 27, 46, 0.8)",
            paper_bgcolor="rgba(19, 27, 46, 0.8)",
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        fig_age.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption("Younger borrowers (<30) exhibit higher default frequency (~11.5%) compared to mature cohorts (60+: ~5.0%).")

    with row2_col2:
        st.markdown("#### 4. Annuity-to-Income / Debt-to-Income (DTI) Ratio")
        df_dti = loader.execute_query("""
            SELECT 
                CASE 
                    WHEN (AMT_ANNUITY / (AMT_INCOME_TOTAL + 1e-5)) < 0.15 THEN 'Low DTI (<15%)'
                    WHEN (AMT_ANNUITY / (AMT_INCOME_TOTAL + 1e-5)) BETWEEN 0.15 AND 0.30 THEN 'Moderate DTI (15-30%)'
                    WHEN (AMT_ANNUITY / (AMT_INCOME_TOTAL + 1e-5)) BETWEEN 0.30 AND 0.45 THEN 'High DTI (30-45%)'
                    ELSE 'Critical DTI (>45%)'
                END AS dti_tier,
                COUNT(*) AS applicants,
                ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct
            FROM loan_applications
            WHERE AMT_INCOME_TOTAL > 0 AND AMT_ANNUITY > 0
            GROUP BY dti_tier
            ORDER BY default_rate_pct ASC;
        """)
        fig_dti = px.bar(
            df_dti, x="dti_tier", y="default_rate_pct",
            text="default_rate_pct",
            color="default_rate_pct",
            color_continuous_scale=["#10B981", "#EF4444"],
            labels={"dti_tier": "Debt-to-Income Tier", "default_rate_pct": "Default Rate (%)"}
        )
        fig_dti.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(19, 27, 46, 0.8)",
            paper_bgcolor="rgba(19, 27, 46, 0.8)",
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        fig_dti.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig_dti, use_container_width=True)
        st.caption("Applicants whose annuity payments exceed 30% of income enter high-risk delinquency territory.")

    # Insight 5
    st.markdown("#### 5. High-Risk Occupation Ranking (Sample Size >= 500)")
    df_occ = loader.execute_query("""
        SELECT 
            COALESCE(OCCUPATION_TYPE, 'Unspecified') AS occupation,
            COUNT(*) AS total_applicants,
            SUM(TARGET) AS total_defaulters,
            ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
            ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_annual_income
        FROM loan_applications
        WHERE OCCUPATION_TYPE IS NOT NULL
        GROUP BY OCCUPATION_TYPE
        HAVING COUNT(*) >= 500
        ORDER BY default_rate_pct DESC;
    """)
    fig_occ = px.bar(
        df_occ, x="default_rate_pct", y="occupation",
        orientation="h",
        text="default_rate_pct",
        color="default_rate_pct",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        labels={"occupation": "Occupation Type", "default_rate_pct": "Default Rate (%)"}
    )
    fig_occ.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(19, 27, 46, 0.8)",
        paper_bgcolor="rgba(19, 27, 46, 0.8)",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_showscale=False
    )
    fig_occ.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_occ, use_container_width=True)

# ==============================================================================
# TAB 2: REAL-TIME UNDERWRITING & APPLICANT SCORER
# ==============================================================================
with tab2:
    st.markdown("### Real-Time Applicant Underwriting & Credit Scoring Engine")
    st.markdown("Evaluate individual credit applications with instant calibrated risk scores, Basel III risk tiers, and SHAP feature attributions.")

    # Quick Presets for Instant Testing
    st.markdown("#### Quick Applicant Profile Presets:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    preset_choice = None
    with p_col1:
        if st.button("Preset: Prime Borrower (Low Risk)", use_container_width=True):
            preset_choice = "prime"
    with p_col2:
        if st.button("Preset: Moderate Risk Borrower", use_container_width=True):
            preset_choice = "moderate"
    with p_col3:
        if st.button("Preset: Subprime / Stressed Borrower", use_container_width=True):
            preset_choice = "subprime"
    with p_col4:
        if st.button("Reset Defaults", use_container_width=True):
            preset_choice = "default"

    # Default preset values
    val_income = 180000.0
    val_credit = 500000.0
    val_annuity = 25000.0
    val_goods = 450000.0
    val_age = 38
    val_emp = 6.5
    val_ext1 = 0.55
    val_ext2 = 0.62
    val_ext3 = 0.58
    val_edu = "Higher education"
    val_inc_type = "Working"
    val_fam = "Married"
    val_occ = "Core staff"

    if preset_choice == "prime":
        val_income = 250000.0
        val_credit = 400000.0
        val_annuity = 18000.0
        val_goods = 380000.0
        val_age = 45
        val_emp = 12.0
        val_ext1 = 0.78
        val_ext2 = 0.82
        val_ext3 = 0.85
        val_edu = "Higher education"
        val_inc_type = "Commercial associate"
        val_fam = "Married"
        val_occ = "Managers"
    elif preset_choice == "moderate":
        val_income = 135000.0
        val_credit = 450000.0
        val_annuity = 24000.0
        val_goods = 400000.0
        val_age = 32
        val_emp = 3.5
        val_ext1 = 0.45
        val_ext2 = 0.48
        val_ext3 = 0.50
        val_edu = "Secondary / secondary special"
        val_inc_type = "Working"
        val_fam = "Single / not married"
        val_occ = "Sales staff"
    elif preset_choice == "subprime":
        val_income = 75000.0
        val_credit = 600000.0
        val_annuity = 38000.0
        val_goods = 550000.0
        val_age = 23
        val_emp = 0.8
        val_ext1 = 0.18
        val_ext2 = 0.22
        val_ext3 = 0.15
        val_edu = "Lower secondary"
        val_inc_type = "Working"
        val_fam = "Single / not married"
        val_occ = "Laborers"

    col_inputs, col_results = st.columns([1, 1.2])

    with col_inputs:
        st.markdown("#### Financial & Loan Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            amt_income = st.number_input("Annual Income ($)", min_value=10000.0, max_value=5000000.0, value=val_income, step=5000.0)
            amt_credit = st.number_input("Requested Credit ($)", min_value=20000.0, max_value=4000000.0, value=val_credit, step=10000.0)
            amt_annuity = st.number_input("Monthly Annuity ($)", min_value=1000.0, max_value=300000.0, value=val_annuity, step=1000.0)
            age_years = st.slider("Applicant Age (Years)", min_value=20, max_value=75, value=int(val_age))
            years_employed = st.slider("Years Employed", min_value=0.0, max_value=40.0, value=float(val_emp), step=0.5)

        with c2:
            amt_goods = st.number_input("Goods Price ($)", min_value=10000.0, max_value=4000000.0, value=val_goods, step=10000.0)
            education = st.selectbox("Education Level", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"], index=["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"].index(val_edu) if val_edu in ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"] else 0)
            income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "State servant", "Pensioner"], index=["Working", "Commercial associate", "State servant", "Pensioner"].index(val_inc_type) if val_inc_type in ["Working", "Commercial associate", "State servant", "Pensioner"] else 0)
            family_status = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], index=["Married", "Single / not married", "Civil marriage", "Separated", "Widow"].index(val_fam) if val_fam in ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"] else 0)
            occupation = st.selectbox("Occupation", ["Core staff", "Managers", "Laborers", "Sales staff", "Drivers", "Accountants", "High skill tech staff"], index=["Core staff", "Managers", "Laborers", "Sales staff", "Drivers", "Accountants", "High skill tech staff"].index(val_occ) if val_occ in ["Core staff", "Managers", "Laborers", "Sales staff", "Drivers", "Accountants", "High skill tech staff"] else 0)

        st.markdown("#### External Bureau Ratings (0.0 to 1.0)")
        b1, b2, b3 = st.columns(3)
        with b1:
            ext_1 = st.slider("Agency 1 Score", 0.0, 1.0, float(val_ext1), 0.01)
        with b2:
            ext_2 = st.slider("Agency 2 Score", 0.0, 1.0, float(val_ext2), 0.01)
        with b3:
            ext_3 = st.slider("Agency 3 Score", 0.0, 1.0, float(val_ext3), 0.01)

        c_f1, c_f2 = st.columns(2)
        with c_f1:
            flag_car = st.radio("Vehicle Ownership", ["Y", "N"], horizontal=True)
        with c_f2:
            flag_realty = st.radio("Real Estate Ownership", ["Y", "N"], horizontal=True)

        # Build applicant row
        applicant_payload = {
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "AMT_GOODS_PRICE": amt_goods,
            "DAYS_BIRTH": -int(age_years * 365.25),
            "DAYS_EMPLOYED": -int(years_employed * 365.25),
            "NAME_EDUCATION_TYPE": education,
            "NAME_INCOME_TYPE": income_type,
            "NAME_FAMILY_STATUS": family_status,
            "OCCUPATION_TYPE": occupation,
            "EXT_SOURCE_1": ext_1,
            "EXT_SOURCE_2": ext_2,
            "EXT_SOURCE_3": ext_3,
            "FLAG_OWN_CAR": flag_car,
            "FLAG_OWN_REALTY": flag_realty,
            "CNT_FAM_MEMBERS": 2.0,
            "REGION_RATING_CLIENT": 2
        }

    with col_results:
        st.markdown("#### Quantitative Underwriting Decision")
        
        # Run inference
        pred_res = predictor.predict_applicant(applicant_payload)
        score = pred_res["credit_score"]
        prob = pred_res["default_probability"]
        prob_pct = pred_res["default_probability_pct"]
        risk_band = pred_res["risk_band"]
        recommendation = pred_res["recommendation"]
        badge_text = pred_res["badge_text"]
        color = pred_res["theme_color"]

        # Gauge Chart (300 to 850)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"FICO-Scaled Credit Score: {score} / 850", 'font': {'size': 18, 'color': '#FFFFFF', 'family': 'Plus Jakarta Sans'}},
            number={'font': {'size': 38, 'color': color, 'family': 'JetBrains Mono'}},
            gauge={
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': color, 'thickness': 0.28},
                'bgcolor': "#131B2E",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [300, 600], 'color': 'rgba(239, 68, 68, 0.22)'},
                    {'range': [600, 750], 'color': 'rgba(245, 158, 11, 0.22)'},
                    {'range': [750, 850], 'color': 'rgba(16, 185, 129, 0.22)'}
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 3},
                    'thickness': 0.8,
                    'value': score
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(19, 27, 46, 0.8)",
            paper_bgcolor="rgba(19, 27, 46, 0.8)",
            height=250,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Underwriting Decision Banner
        if risk_band == "LOW RISK":
            badge_class = "badge-auto-approve"
        elif risk_band == "MEDIUM RISK":
            badge_class = "badge-manual-review"
        else:
            badge_class = "badge-decline"

        # Key domain ratios for this applicant
        dti_pct = round((amt_annuity / (amt_income + 1e-5)) * 100, 1)
        loan_inc_ratio = round(amt_credit / (amt_income + 1e-5), 2)
        comp_bureau = round((ext_1 + ext_2 + ext_3) / 3.0, 2)

        st.markdown(f"""
        <div style="background: linear-gradient(180deg, #131B2E 0%, #0F172A 100%); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 18px 20px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="{badge_class}">{badge_text}</span>
                <span style="font-size: 0.92rem; color: #94A3B8;">Predicted Default Probability: <b style="color: #F8FAFC; font-family: 'JetBrains Mono';">{prob_pct}%</b></span>
            </div>
            <div style="font-size: 0.95rem; color: #E2E8F0; line-height: 1.5; margin-bottom: 14px;">{recommendation}</div>
            <div style="display: flex; gap: 12px; border-top: 1px solid #334155; padding-top: 12px; font-size: 0.82rem; color: #94A3B8;">
                <div>DTI Ratio: <b style="color: {'#EF4444' if dti_pct > 30 else '#10B981'}; font-family: 'JetBrains Mono';">{dti_pct}%</b></div>
                <div>Credit/Income: <b style="color: #F8FAFC; font-family: 'JetBrains Mono';">{loan_inc_ratio}x</b></div>
                <div>Composite Bureau: <b style="color: #38BDF8; font-family: 'JetBrains Mono';">{comp_bureau}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP Explainability & Narrative Breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Explainable AI: SHAP Factor Attribution & Underwriter Narrative")
    
    try:
        app_df = pd.DataFrame([applicant_payload])
        explanation = explainer.explain_applicant(app_df)
        
        xai_col1, xai_col2 = st.columns([1.2, 1])
        
        with xai_col1:
            st.markdown("#### Local Feature Attribution (SHAP Waterfall Components)")
            top_features = explanation["contributions_top10"]
            df_shap_plot = pd.DataFrame(top_features)
            
            # Plotly horizontal bar for SHAP values
            df_shap_plot["color"] = df_shap_plot["shap_value"].apply(lambda x: "#EF4444" if x > 0 else "#10B981")
            
            fig_shap = px.bar(
                df_shap_plot,
                x="shap_value",
                y="display_name",
                orientation="h",
                color="color",
                color_discrete_map="identity",
                labels={"shap_value": "SHAP Impact on Default Log-Odds (Red = Risk Driver, Green = Safety Driver)", "display_name": "Risk Feature"}
            )
            fig_shap.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(19, 27, 46, 0.8)",
                paper_bgcolor="rgba(19, 27, 46, 0.8)",
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        with xai_col2:
            st.markdown("#### Plain-English Underwriter Credit Memo")
            bullets = explanation["underwriter_summary_bullets"]
            
            memo_html = "<div class='narrative-card'>"
            for b in bullets:
                clean_b = b.replace("**", "<b>").replace("**", "</b>")
                if clean_b.startswith("•"):
                    memo_html += f"<div style='margin-bottom: 8px; font-size: 0.88rem; color: #CBD5E1; line-height: 1.4;'>{clean_b}</div>"
                else:
                    memo_html += f"<div style='margin-top: 10px; margin-bottom: 6px; font-size: 0.95rem; font-weight: 700; color: #38BDF8;'>{clean_b}</div>"
            memo_html += "</div>"
            
            st.markdown(memo_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error computing SHAP explanation: {e}")

# ==============================================================================
# TAB 3: CREDIT POLICY RULE ENGINE
# ==============================================================================
with tab3:
    st.markdown("### Transparent Credit Policy & Decision Tree Rule Engine")
    st.markdown("Extracted decision tree induction rules provide transparent, deterministic credit guidelines for underwriting committees.")

    # Load / Extract Rules
    rules_list = rule_engine.fit_and_extract_rules() if not rule_engine.rules else rule_engine.rules

    r_col1, r_col2 = st.columns([1, 2])
    with r_col1:
        tier_filter = st.selectbox("Filter Rules by Risk Tier", ["All Rules", "LOW RISK", "MEDIUM RISK", "HIGH RISK"])
    with r_col2:
        search_kw = st.text_input("Search rule conditions (e.g. EXT_SOURCES, ANNUITY)", value="")

    filtered_rules = rules_list
    if tier_filter != "All Rules":
        filtered_rules = [r for r in filtered_rules if r["assigned_tier"] == tier_filter]
    if search_kw:
        filtered_rules = [r for r in filtered_rules if search_kw.lower() in r["condition"].lower()]

    # Display Rules Table
    rule_rows = []
    for r in filtered_rules:
        rule_rows.append({
            "Rule ID": r["rule_id"],
            "Risk Tier": r["assigned_tier"],
            "Default Rate": f"{r['empirical_default_rate_pct']}%",
            "Population Coverage": f"{r['leaf_samples']:,} applicants",
            "Recommended Action": r["recommended_action"],
            "Rule Logic Condition": r["condition"]
        })

    df_rules_display = pd.DataFrame(rule_rows)
    st.dataframe(df_rules_display, use_container_width=True, hide_index=True)

    # Download Rules Button
    st.download_button(
        "Export Policy Ruleset (JSON)",
        data=json.dumps(rules_list, indent=2),
        file_name="credit_policy_rules.json",
        mime="application/json"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Active Applicant Rule Matching")
    
    # Check current applicant from Tab 2
    matched = rule_engine.evaluate_applicant_rules(applicant_payload)
    if matched:
        st.success(f"Applicant matched {len(matched)} credit policy rule(s):")
        for m in matched:
            st.markdown(f"""
            - **{m['rule_id']}** (`{m['assigned_tier']}` | Default Rate: `{m['empirical_default_rate_pct']}%`): {m['recommended_action']}
              - *Condition*: `{m['condition']}`
            """)
    else:
        st.info("Applicant falls within standard baseline underwriting bounds.")

# ==============================================================================
# TAB 4: TALK-TO-DATA AI COPILOT
# ==============================================================================
with tab4:
    st.markdown("### Talk-to-Data Natural Language to SQL Copilot")
    st.markdown("Ask natural language analytical questions across 307,511 loan records. The copilot generates AST-sanitized DuckDB SQL and delivers sub-15ms aggregations.")

    # Pre-canned prompt buttons
    st.markdown("#### Suggested Inquiries:")
    c_q1, c_q2, c_q3, c_q4 = st.columns(4)
    
    selected_query = None
    with c_q1:
        if st.button("Default rate by income bracket", use_container_width=True):
            selected_query = "What is the default rate across different annual income brackets (<$50k, $50k-$100k, >$100k)?"
    with c_q2:
        if st.button("Top 5 highest risk occupations", use_container_width=True):
            selected_query = "Show top 5 highest risk occupations with at least 500 applicants."
    with c_q3:
        if st.button("Car owners vs Non-car owners", use_container_width=True):
            selected_query = "Compare default rate and loan amount between car owners and non-car owners."
    with c_q4:
        if st.button("Risk profile across age cohorts", use_container_width=True):
            selected_query = "What is the risk profile across age cohorts (under 30, 30s, 40s, 50s, 60+)?"

    # Query Input Box
    user_query = st.text_input(
        "Enter your financial question in natural language:",
        value=selected_query or "Show default rate by education level and average credit amount",
        placeholder="e.g. Compare default rate between married and single applicants..."
    )

    if st.button("Execute Intelligence Query", type="primary"):
        if user_query:
            with st.spinner("Translating natural language to DuckDB SQL..."):
                res = nl_agent.process_query(user_query)

            if res["success"]:
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Execution Badge
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="badge-latency">{res['latency_badge']}</span>
                    <span style="font-size: 0.85rem; color: #94A3B8;">Records Returned: <b style="color: #F8FAFC;">{res['row_count']}</b></span>
                </div>
                """, unsafe_allow_html=True)

                # SQL Code Viewer
                with st.expander("Generated DuckDB SQL Query (AST Validated)", expanded=True):
                    st.code(res["sql"], language="sql")

                # Results Data & Charts
                df_res = res["data"]
                if df_res is not None and not df_res.empty:
                    col_t1, col_t2 = st.columns([1.2, 1])

                    with col_t1:
                        st.markdown("#### Query Result Matrix")
                        st.dataframe(df_res, use_container_width=True)

                    with col_t2:
                        st.markdown("#### Executive Synthesis")
                        st.markdown("<div class='narrative-card'>", unsafe_allow_html=True)
                        for bullet in res["executive_summary"]:
                            st.markdown(f"- {bullet}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Auto Charting if 2+ columns
                    if len(df_res.columns) >= 2 and len(df_res) > 1:
                        first_col = df_res.columns[0]
                        numeric_cols = [c for c in df_res.columns[1:] if np.issubdtype(df_res[c].dtype, np.number)]
                        
                        if numeric_cols:
                            st.markdown("#### Interactive Visualization")
                            target_col = numeric_cols[0]
                            fig_dyn = px.bar(
                                df_res, x=first_col, y=target_col,
                                text=target_col,
                                color=target_col,
                                color_continuous_scale="Blues",
                                labels={first_col: first_col.replace("_", " ").title(), target_col: target_col.replace("_", " ").title()}
                            )
                            fig_dyn.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(19, 27, 46, 0.8)",
                                paper_bgcolor="rgba(19, 27, 46, 0.8)",
                                margin=dict(l=20, r=20, t=30, b=20),
                                coloraxis_showscale=False
                            )
                            fig_dyn.update_traces(textposition='outside')
                            st.plotly_chart(fig_dyn, use_container_width=True)
            else:
                st.error(f"Query Execution Error: {res['error']}")

# ==============================================================================
# SIDEBAR DIAGNOSTICS & SYSTEM STATUS
# ==============================================================================
with st.sidebar:
    st.markdown("### System & Model Telemetry")
    health = get_system_health()
    st.markdown(f"""
    <div class="sidebar-telemetry">
        <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 4px;">HOST PLATFORM</div>
        <div style="font-weight: 700; font-size: 0.95rem; color: #FFFFFF; font-family: 'JetBrains Mono';">{health['platform']} (Python {health['python_version']})</div>
        <hr style="border: none; border-top: 1px solid #334155; margin: 8px 0;" />
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #94A3B8;">
            <span>CPU Usage:</span>
            <span style="color: #38BDF8; font-family: 'JetBrains Mono';">{health['cpu_usage_pct']}%</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #94A3B8;">
            <span>Memory Active:</span>
            <span style="color: #38BDF8; font-family: 'JetBrains Mono';">{health['memory_used_mb']} MB</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #94A3B8;">
            <span>Storage Free:</span>
            <span style="color: #38BDF8; font-family: 'JetBrains Mono';">{health['disk_free_gb']} GB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Machine Learning Models")
    st.markdown("""
    - **Champion Model**: `LightGBM GBDT`
      - Imbalance: `scale_pos_weight=11.387`
      - Validation: `Stratified 5-Fold CV`
    - **Baseline Model**: `Logistic Regression`
      - Imbalance: `class_weight='balanced'`
    """)

    st.markdown("---")
    st.markdown("### LLM Agent Provider")
    active_prov = Config.LLM_PROVIDER.upper()
    st.markdown(f"Active Provider: **{active_prov}**")
    if Config.GROQ_API_KEY:
        st.caption("Groq API Key: Configured (llama-3.3-70b-versatile)")
    elif Config.GEMINI_API_KEY:
        st.caption("Gemini API Key: Configured")
    elif Config.OPENAI_API_KEY:
        st.caption("OpenAI API Key: Configured")
    else:
        st.caption("Mode: Deterministic Offline Synthesizer")

    st.markdown("---")
    st.caption("Enterprise Credit Risk Intelligence Platform (CRIP)")
