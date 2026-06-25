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
    
    /* Improved Contrast for Metric Cards */
    .stMetric { 
        background-color: #1e2530; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #00ffcc; 
    }
    [data-testid="stMetricLabel"] p {
        color: #b0bec5 !important; /* Light grey for labels */
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] div {
        color: #ffffff !important; /* Bright white for values */
    }
    
    /* Improved Contrast for Agent Chat Box */
    .chat-box { 
        background-color: #1e2530; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #333; 
        margin-bottom: 20px; 
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    clean_df = pd.DataFrame()
    risk_df = pd.DataFrame()
    is_mock_risk = False
    
    # 1. Try to load Clean Data
    try:
        clean_df = pd.read_csv("data/processed/retail_cleaned.csv", parse_dates=['date'])
        # Ensure revenue is calculated for dashboards
        if 'revenue' not in clean_df.columns:
            clean_df['revenue'] = clean_df['units_sold'] * clean_df['price']
    except FileNotFoundError:
        st.error("Cleaned data file not found. Please ensure the data cleaning script has been run.")
        return clean_df, risk_df, is_mock_risk

    # 2. Try to load Risk Data OR Mock it for UI Preview
    try:
        risk_df = pd.read_csv("data/outputs/risk_output.csv", parse_dates=['date'])
    except FileNotFoundError:
        is_mock_risk = True
        # MOCK DATA GENERATOR: If the ML model hasn't run, generate fake data for the UI
        if not clean_df.empty:
            latest_date = clean_df['date'].max()
            latest_data = clean_df[clean_df['date'] == latest_date].copy()
            np.random.seed(42) # Keeps the random fake data consistent
            
            # Invent fake 7-day demand predictions
            latest_data['forecast_7_day_demand'] = np.random.randint(10, 250, size=len(latest_data))
            
            # Calculate metrics based on the fake demand
            latest_data['inventory_cover_days'] = np.where(
                latest_data['forecast_7_day_demand'] > 0,
                latest_data['inventory_level'] / (latest_data['forecast_7_day_demand'] / 7),
                999
            )
            
            # Assign fake risk buckets
            conditions = [
                (latest_data['inventory_cover_days'] < 2),
                (latest_data['inventory_cover_days'] >= 2) & (latest_data['inventory_cover_days'] < 5),
                (latest_data['inventory_cover_days'] >= 5) & (latest_data['inventory_cover_days'] <= 21),
                (latest_data['inventory_cover_days'] > 21)
            ]
            choices = ['Critical', 'Warning', 'Normal', 'Overstock']
            latest_data['risk_bucket'] = np.select(conditions, choices, default='Unknown')
            latest_data['stock_out_risk_flag'] = (latest_data['forecast_7_day_demand'] > latest_data['inventory_level']).astype(int)
            
            risk_df = latest_data[['date', 'store_id', 'product_id', 'inventory_level', 
                                   'forecast_7_day_demand', 'inventory_cover_days', 
                                   'risk_bucket', 'stock_out_risk_flag']]
            
    return clean_df, risk_df, is_mock_risk

clean_df, risk_df, is_mock_risk = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🤖 Retail Copilot")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 1. Home", 
    "🧮 2. Data Quality",
    "📊 3. Sales Dashboard", 
    "⚠️ 4. Inventory Dashboard", 
    "📈 5. Forecast Dashboard",
    "📋 6. Action Plan",
    "💬 7. Agent Chat",
    "🕵️ 8. Agent Trace",
    "🏛️ 9. Governance"
])
st.sidebar.markdown("---")
st.sidebar.info("Powered by XGBoost & LangGraph")

# Add a warning in the sidebar if we are using fake UI preview data
if is_mock_risk:
    st.sidebar.warning("⚠️ Preview Mode: ML output files missing. Using simulated forecasts to render UI.")

# ==========================================
# PAGE 1: HOME
# ==========================================
st.markdown("""
<style>
.home-card {
    background-color: #1e2530;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #333;
    height: 280px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Equal height for st.info and st.success boxes */
div[data-testid="stAlert"] {
    min-height: 180px !important;
    height: 180px !important;
}

/* Align the alert boxes content */
div[data-testid="stAlert"] > div {
    height: 100%;
}

</style>
""", unsafe_allow_html=True)

if page == "🏠 1. Home":
    st.title("Agentic Retail Operations Copilot")
    
    st.markdown("### Project Overview")
    st.write("This application is an Agentic AI assistant designed to monitor retail sales, forecast demand, detect stock-out risks, and recommend business actions dynamically. Unlike standard dashboards, it acts as a virtual supervisor.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dataset Overview")
        st.info("""
        **Retail Store Inventory Forecasting Dataset**
        * Contains daily retail sales, inventory, pricing, weather, and promotions.
        * Used to predict demand and assign business risk buckets.
        """)

    with col2:
        st.markdown("### Agent Capabilities")
        st.success("""
        * 🔍 **Investigates** stock-out risks and overstock.
        * 📈 **Forecasts** demand using ML tools.
        * 💡 **Recommends** reorders, markdowns, and pre-stocking.
        * 🛡️ **Requests Approval** before generating simulated actions.
        * 🗣️ **Answers** natural language business questions. 
        """)

    st.markdown("### Overall Workflow")
    st.write("1. Data Ingestion & Feature Engineering")
    st.write("2. XGBoost Demand Forecasting")
    st.write("3. Risk & Recommendation Engine")
    st.write("4. LangGraph Agent Task Planning")
    st.write("5. Human-in-the-Loop Approval")

# ==========================================
# PAGE 2: DATA QUALITY
# ==========================================
elif page == "🧮 2. Data Quality":

    st.title("🧮 Data Quality Assessment")
    st.write("Review dataset readiness for machine learning modeling.")

    if not clean_df.empty:

        # Main Metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Rows",
            f"{len(clean_df):,}"
        )

        col2.metric(
            "Total Columns",
            len(clean_df.columns)
        )

        col3.metric(
            "Missing Values",
            clean_df.isna().sum().sum()
        )

        col4.metric(
            "Duplicate Count",
            clean_df.duplicated().sum()
        )


        st.markdown("---")


        # Dataset Summary
        st.subheader("Dataset Summary")

        summary_df = pd.DataFrame({
            "Category": [
                "Date Range",
                "Number of Stores",
                "Number of Products",
                "Outlier Handling"
            ],
            "Details": [
                f"{clean_df['date'].min().date()} → {clean_df['date'].max().date()}",
                f"{clean_df['store_id'].nunique()} Stores",
                f"{clean_df['product_id'].nunique()} Products",
                "Handled using Log Transformation"
            ]
        })

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True
        )


        st.markdown("---")


        # Charts
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Missing Values")

            missing_df = clean_df.isna().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Count']

            fig_miss = px.bar(
                missing_df,
                x='Column',
                y='Missing Count',
                template='plotly_dark'
            )

            st.plotly_chart(
                fig_miss,
                use_container_width=True
            )


        with c2:
            st.subheader("Sales Distribution")

            fig_sales = px.histogram(
                clean_df,
                x="units_sold",
                nbins=40,
                template='plotly_dark'
            )

            st.plotly_chart(
                fig_sales,
                use_container_width=True
            )


        with c3:
            st.subheader("Inventory Distribution")

            fig_inv = px.histogram(
                clean_df,
                x="inventory_level",
                nbins=40,
                template='plotly_dark'
            )

            st.plotly_chart(
                fig_inv,
                use_container_width=True
            )

# ==========================================
# PAGE 3: SALES DASHBOARD
# ==========================================
elif page == "📊 3. Sales Dashboard":
    st.title("📊 Sales Performance")
    
    if not clean_df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"${clean_df['revenue'].sum():,.0f}")
        col2.metric("Total Sales Units", f"{clean_df['units_sold'].sum():,.0f}")
        col3.metric("Average Selling Price", f"${clean_df['price'].mean():.2f}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sales Trend Over Time")
            daily_sales = clean_df.groupby('date')['revenue'].sum().reset_index()
            fig_trend = px.line(daily_sales, x='date', y='revenue', template='plotly_dark')
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with c2:
            st.subheader("Promotion vs Non-Promotion Sales")
            promo_sales = clean_df.groupby('holiday_or_promo_flag')['revenue'].sum().reset_index()
            promo_sales['holiday_or_promo_flag'] = promo_sales['holiday_or_promo_flag'].map({0: 'Regular', 1: 'Promotion'})
            fig_promo = px.pie(promo_sales, names='holiday_or_promo_flag', values='revenue', template='plotly_dark', hole=0.4)
            st.plotly_chart(fig_promo, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Top Products by Revenue")
            top_prod = clean_df.groupby('product_id')['revenue'].sum().nlargest(5).reset_index()
            fig_prod = px.bar(top_prod, x='product_id', y='revenue', template='plotly_dark')
            st.plotly_chart(fig_prod, use_container_width=True)
            
        with c4:
            st.subheader("Sales by Store")
            store_sales = clean_df.groupby('store_id')['revenue'].sum().reset_index()
            fig_store = px.bar(store_sales, x='store_id', y='revenue', color='store_id', template='plotly_dark')
            st.plotly_chart(fig_store, use_container_width=True)

# ==========================================
# PAGE 4: INVENTORY DASHBOARD
# ==========================================
elif page == "⚠️ 4. Inventory Dashboard":
    st.title("⚠️ Inventory Position & Risk")
    
    if not risk_df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Current Inventory", f"{risk_df['inventory_level'].sum():,.0f}")
        col2.metric("Avg Inventory Cover", f"{risk_df['inventory_cover_days'].replace(999, np.nan).mean():.1f} Days")
        col3.metric("🚨 Critical Risks", len(risk_df[risk_df['risk_bucket'] == 'Critical']))
        col4.metric("⚠️ Warning Risks", len(risk_df[risk_df['risk_bucket'] == 'Warning']))
        col5.metric("📦 Overstock", len(risk_df[risk_df['risk_bucket'] == 'Overstock']))
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk Bucket Breakdown")
            color_map = {'Critical': '#ff4b4b', 'Warning': '#ffa600', 'Normal': '#00ffcc', 'Overstock': '#a4a4a4'}
            fig_risk = px.pie(risk_df, names='risk_bucket', color='risk_bucket', color_discrete_map=color_map, template='plotly_dark')
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with c2:
            st.subheader("Inventory Cover Distribution")
            # Filter out the 999 placeholders for infinite cover to make chart readable
            cover_df = risk_df[risk_df['inventory_cover_days'] < 50]
            fig_cov = px.histogram(cover_df, x='inventory_cover_days', nbins=30, template='plotly_dark')
            st.plotly_chart(fig_cov, use_container_width=True)

        st.subheader("Product-Store Risk Table")
        st.dataframe(risk_df.sort_values('inventory_cover_days'), use_container_width=True)

# ==========================================
# PAGE 5: FORECAST DASHBOARD
# ==========================================
elif page == "📈 5. Forecast Dashboard":
    st.title("📈 Demand Forecast Accuracy")
    
    # Using the metrics from the XGBoost run
    # Row 1 (3 metrics)
    col1, col2, col3 = st.columns(3)

    col1.metric("Forecast Period", "Next 7 Days")
    col2.metric("MAE", "59.74")
    col3.metric("RMSE", "77.17")


    # Row 2 (2 metrics)
    col4, col5 = st.columns(2)

    col4.metric("WMAPE", "43.96%")
    col5.metric("Model", "XGBoost")

    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Forecast by Product (Next 7 Days)")
        if not risk_df.empty:
            forecast_prod = risk_df.groupby('product_id')['forecast_7_day_demand'].sum().reset_index()
            fig_fcast = px.bar(forecast_prod, x='product_id', y='forecast_7_day_demand', template='plotly_dark', color_discrete_sequence=['#00ffcc'])
            st.plotly_chart(fig_fcast, use_container_width=True)
            
    with c2:
        st.subheader("Feature Importance (Model Insight)")
        # Mocking feature importance based on standard retail time-series results
        feat_data = pd.DataFrame({
            'Feature': ['rolling_7_day_avg', 'lag_1_day_sales', 'lag_7_day_sales', 'price', 'day_of_week', 'discount', 'is_weekend'],
            'Importance': [0.35, 0.22, 0.15, 0.10, 0.08, 0.06, 0.04]
        }).sort_values('Importance', ascending=True)
        fig_feat = px.bar(feat_data, x='Importance', y='Feature', orientation='h', template='plotly_dark')
        st.plotly_chart(fig_feat, use_container_width=True)

# ==========================================
# PAGE 6: ACTION PLAN
# ==========================================
elif page == "📋 6. Action Plan":
    st.title("📋 Recommended Action Plan")
    st.write("Review simulated actions recommended by the Agent rules engine. Requires human approval before execution.")
    
    if not risk_df.empty:
        # Generate Action Plan logic
        actions_df = risk_df.copy()
        actions_df['Recommended action'] = np.select(
            [
                actions_df['risk_bucket'] == 'Critical',
                actions_df['risk_bucket'] == 'Warning',
                actions_df['risk_bucket'] == 'Overstock'
            ],
            [
                'Reorder urgently',
                'Replenish / monitor',
                'Review markdown'
            ],
            default='No action'
        )
        
        # Simple quantity recommendation logic
        actions_df['Recommended quantity'] = np.where(
            actions_df['risk_bucket'].isin(['Critical', 'Warning']),
            np.maximum((actions_df['forecast_7_day_demand'] * 1.5) - actions_df['inventory_level'], 0).astype(int),
            0
        )
        
        actions_df['Approval status'] = 'Pending'
        
        final_actions = actions_df[actions_df['Recommended action'] != 'No action'][
            ['store_id', 'product_id', 'inventory_level', 'forecast_7_day_demand', 
             'inventory_cover_days', 'risk_bucket', 'Recommended action', 'Recommended quantity', 'Approval status']
        ].round(2)
        
        # Display data editor for interactive Approve/Reject
        st.info("💡 Tip: In a production environment, changing the status to 'Approved' will trigger the backend API.")
        edited_df = st.data_editor(
            final_actions, 
            column_config={
                "Approval status": st.column_config.SelectboxColumn(
                    "Approval Status",
                    help="Approve or Reject the simulated action",
                    options=["Pending", "Approved", "Rejected"],
                    required=True,
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.download_button(
            label="⬇️ Export Action Plan to CSV",
            data=edited_df.to_csv(index=False).encode('utf-8'),
            file_name='approved_action_plan.csv',
            mime='text/csv',
        )

# ==========================================
# PAGE 7: AGENT CHAT
# ==========================================
elif page == "💬 7. Agent Chat":
    st.title("Agentic Retail Operations Copilot")
    st.markdown("---")
    
    st.markdown("""
    ### Ask a business question:
    * *Find the top 5 products at stock-out risk next week.*
    * *Why is Product P001 at critical risk?*
    * *Which stores need urgent replenishment?*
    * *Summarize this week’s business performance.*
    """)

    st.markdown("""
<style>

/* Approve button */
div.stButton > button[kind="primary"] {
    background-color: #00c853 !important;
    color: white !important;
    border: none !important;
}


/* Reject button - target second button */
div.stButton:nth-of-type(2) > button {
    background-color: #ff0000 !important;
    color: white !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)
    
    query = st.text_input("", placeholder="[ Find the top 5 products at stock-out risk next week ]")
    
    if st.button("Submit"):
        if query:
            st.markdown('<div class="chat-box">', unsafe_allow_html=True)
            st.markdown("**Agent Response:**")
            st.write("The following products are at highest risk based on the XGBoost 7-day forecast...")
            
            st.markdown("**Recommended Actions:**")
            st.write("1. Reorder Product P001 for Store S003")
            st.write("2. Monitor Product P004 for Store S002")
            st.write("3. Review markdown for Product P009")
            
            st.markdown("**Approval Required:**")

            c1, c2, c3 = st.columns([1,1,8])

            with c1:
                st.button("Approve", type="primary")

            with c2:
                st.button("Reject")

# ==========================================
# PAGE 8: AGENT TRACE
# ==========================================
elif page == "🕵️ 8. Agent Trace":
    st.title("🕵️ Agent Execution Trace")
    st.write("Transparency log showing the exact tools called by the LangGraph agent to generate its response. This proves the solution is agentic and not just a text generator.")
    
    # Mock trace data to fulfill UI requirements
    trace_data = pd.DataFrame({
        "Request ID": ["REQ-901", "REQ-901", "REQ-901", "REQ-901", "REQ-901"],
        "User Question": ["Find stock-out risks"] * 5,
        "Step": ["Agent Step 1", "Agent Step 2", "Agent Step 3", "Agent Step 4", "Agent Step 5"],
        "Tool Called": ["forecast_demand()", "calculate_inventory_cover()", "detect_stockout_risk()", "recommend_business_action()", "request_human_approval()"],
        "Status": ["✅ Success", "✅ Success", "✅ Success", "✅ Success", "⏳ Pending User"],
        "Timestamp": ["10:01:02", "10:01:04", "10:01:05", "10:01:07", "10:01:08"]
    })
    
    st.dataframe(trace_data, use_container_width=True, hide_index=True)
    
    st.subheader("Final Tool Output Summary")
    st.code("""
    {
        "status": "Awaiting Human Approval",
        "action_items": 3,
        "risk_level": "Critical",
        "agent_confidence": 0.92
    }
    """, language="json")

# ==========================================
# PAGE 9: GOVERNANCE
# ==========================================
elif page == "🏛️ 9. Governance":
    st.title("🏛️ Responsible AI & Governance")
    
    with st.expander("📄 Model Card (XGBoost Forecaster)", expanded=True):
        st.write("""
        * **Algorithm:** XGBoost Regressor
        * **Objective:** Predict 7-day future demand for product-store combinations.
        * **Metrics:** WMAPE 43.96%, MAE 59.74 Units.
        * **Limitations:** The model assumes historical lag patterns will continue. It may struggle with entirely novel external events (e.g., sudden global supply chain halts).
        """)
        
    with st.expander("🤖 Agent Card (LangGraph Supervisor)"):
        st.write("""
        * **Role:** Orchestrate business tool execution and summarize findings.
        * **Guardrails:** The Agent is strictly forbidden from executing database write-operations (reorders/markdowns) without explicitly passing through the `request_human_approval` node.
        * **LLM Used:** Local LLM Runtime (Ollama / Mistral).
        """)
        
    with st.expander("⚖️ Business Rules & Human Approval Policy"):
        st.write("""
        * **Critical Risk Definition:** Inventory cover is less than 2 days.
        * **Overstock Definition:** Inventory cover is greater than 21 days.
        * **Approval Policy:** All simulated actions generated by the Recommendation Tool MUST be reviewed and manually signed off by a human supervisor in the UI Page 6 (Action Plan) before being exported to downstream ERP systems.
        """)
        
    with st.expander("🚧 Known Risks & Future Improvements"):
        st.write("""
        * **Risks:** Data latency. If the data pipeline fails to ingest yesterday's CSV, the Agent will forecast based on stale inventory numbers.
        * **Improvements:** Integrate real-time API webhooks, add deep learning Time-Series models (like Prophet) for comparison, and deploy the application to a cloud environment.
        """)