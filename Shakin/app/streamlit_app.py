import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic Retail Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Technical Aesthetic ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #00ffcc; font-family: 'Courier New', Courier, monospace; }
    .stMetric { background-color: #1e2530; padding: 15px; border-radius: 5px; border-left: 5px solid #00ffcc; }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
# @st.cache_data ensures the app doesn't reload the CSVs every time you click a button
@st.cache_data
def load_data():
    try:
        clean_df = pd.read_csv("data/processed/retail_cleaned.csv", parse_dates=['date'])
        risk_df = pd.read_csv("data/outputs/risk_output.csv", parse_dates=['date'])
        return clean_df, risk_df
    except FileNotFoundError:
        st.error("Data files not found. Please ensure the data cleaning and inventory risk scripts have been run.")
        return pd.DataFrame(), pd.DataFrame()

clean_df, risk_df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🤖 Retail Copilot")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", 
    "📊 Sales Dashboard", 
    "⚠️ Inventory Risk", 
    "📈 Forecast Review",
    "💬 Agent Chat (WIP)"
])
st.sidebar.markdown("---")
st.sidebar.info("Powered by XGBoost & LangGraph")

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "🏠 Home":
    st.title("Agentic Retail Operations Copilot")
    st.markdown("""
    ### Welcome to the Retail Operations Command Center
    
    This application is an Agentic AI assistant designed to behave like an intelligent retail supervisor. Instead of just showing static data, this Agent actively investigates business issues and recommends actionable steps.
    
    **What this Agent does:**
    * **🔍 Investigates Inventory Risks:** Identifies which products are at critical stock-out risk or are currently overstocked.
    * **📈 Forecasts Demand:** Uses advanced machine learning (XGBoost) to predict the next 7 days of sales for every product and store.
    * **💡 Recommends Actions:** Suggests specific business actions such as urgent reordering, monitoring, or reviewing markdowns.
    * **🛡️ Asks for Human Approval:** Formulates a simulated action plan and requires human sign-off before any execution.
    * **🗣️ Answers Business Questions:** Understands natural language queries to explain *why* a product is at risk and what the store manager should focus on today.
    
    Use the sidebar to navigate through the data dashboards or chat directly with the Agent.
    """)

# ==========================================
# PAGE 2: SALES DASHBOARD
# ==========================================
elif page == "📊 Sales Dashboard":
    st.title("📊 Sales Performance")
    
    if not clean_df.empty:
        # Calculate KPIs
        clean_df['revenue'] = clean_df['units_sold'] * clean_df['price']
        total_revenue = clean_df['revenue'].sum()
        total_units = clean_df['units_sold'].sum()
        avg_price = clean_df['price'].mean()
        
        # Display Top KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"${total_revenue:,.0f}")
        col2.metric("Total Units Sold", f"{total_units:,.0f}")
        col3.metric("Avg Selling Price", f"${avg_price:.2f}")
        
        st.markdown("---")
        
        # Sales Trend Chart
        st.subheader("Revenue Trend Over Time")
        daily_sales = clean_df.groupby('date')['revenue'].sum().reset_index()
        fig_trend = px.line(daily_sales, x='date', y='revenue', template='plotly_dark')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        # Sales by Region
        with col_chart1:
            st.subheader("Revenue by Region")
            region_sales = clean_df.groupby('region')['revenue'].sum().reset_index()
            fig_region = px.bar(region_sales, x='region', y='revenue', color='region', template='plotly_dark')
            st.plotly_chart(fig_region, use_container_width=True)
            
        # Sales by Category
        with col_chart2:
            st.subheader("Revenue by Category")
            cat_sales = clean_df.groupby('category')['revenue'].sum().reset_index()
            fig_cat = px.pie(cat_sales, names='category', values='revenue', template='plotly_dark')
            st.plotly_chart(fig_cat, use_container_width=True)

# ==========================================
# PAGE 3: INVENTORY RISK
# ==========================================
elif page == "⚠️ Inventory Risk":
    st.title("⚠️ Inventory Risk Assessment")
    
    if not risk_df.empty:
        # KPI Calculation
        critical_count = len(risk_df[risk_df['risk_bucket'] == 'Critical'])
        warning_count = len(risk_df[risk_df['risk_bucket'] == 'Warning'])
        overstock_count = len(risk_df[risk_df['risk_bucket'] == 'Overstock'])
        
        # Risk Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Critical Stock-Out Risk", critical_count)
        col2.metric("⚠️ Warning Risk", warning_count)
        col3.metric("📦 Overstock Items", overstock_count)
        
        st.markdown("---")
        
        col_risk1, col_risk2 = st.columns([1, 2])
        
        with col_risk1:
            st.subheader("Risk Distribution")
            risk_counts = risk_df['risk_bucket'].value_counts().reset_index()
            risk_counts.columns = ['risk_bucket', 'count']
            # Color map matching the risk levels
            color_map = {'Critical': '#ff4b4b', 'Warning': '#ffa600', 'Normal': '#00ffcc', 'Overstock': '#a4a4a4'}
            fig_risk = px.pie(risk_counts, names='risk_bucket', values='count', color='risk_bucket', color_discrete_map=color_map, template='plotly_dark')
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with col_risk2:
            st.subheader("High Risk Items (Immediate Action Required)")
            # Filter for Critical items and sort by how quickly they will run out
            critical_items = risk_df[risk_df['risk_bucket'] == 'Critical'].sort_values('inventory_cover_days')
            st.dataframe(
                critical_items[['store_id', 'product_id', 'inventory_level', 'forecast_7_day_demand', 'inventory_cover_days']],
                use_container_width=True,
                hide_index=True
            )

# ==========================================
# PAGE 4: FORECAST REVIEW
# ==========================================
elif page == "📈 Forecast Review":
    st.title("📈 Demand Forecast Overview")
    st.info("The forecasting engine uses XGBoost to predict the upcoming 7 days of demand based on historical lags, rolling averages, and pricing features.")
    
    if not risk_df.empty:
        st.subheader("Latest 7-Day Forecast Distribution")
        fig_hist = px.histogram(risk_df, x="forecast_7_day_demand", nbins=50, title="Distribution of Forecasted Demand", template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.subheader("Raw Forecast Data")
        st.dataframe(risk_df, use_container_width=True)

# ==========================================
# PAGE 5: AGENT CHAT (PLACEHOLDER)
# ==========================================
elif page == "💬 Agent Chat (WIP)":
    st.title("💬 Retail Supervisor Agent")
    st.markdown("""
    > **Status:** Under Construction 🚧
    
    This page will house the LangGraph-powered AI Agent. 
    Once completed, you will be able to ask natural language questions like:
    
    * *"Find the top 5 products at stock-out risk next week and recommend actions."*
    * *"Why is Product P001 at critical risk?"*
    * *"Which stores need urgent replenishment?"*
    
    The Agent will decide which tools (like the Forecasting or Risk tools we just built) to call, analyze the data, and return a summary and action plan for human approval.
    """)
    
    st.text_input("Ask the agent a question (Disabled):", disabled=True)
    st.button("Send", disabled=True)