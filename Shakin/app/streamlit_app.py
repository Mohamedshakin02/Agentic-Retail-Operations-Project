import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Fix Python Path for Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(os.path.join(root_dir, "src"))

# --- Backend Functionality Imports ---
try:
    from config import CLEANED_DATA_PATH, RISK_OUTPUT_PATH, FORECAST_OUTPUT_PATH
    from data_quality import generate_quality_report
    from approval import submit_approval_decision, save_approved_action, get_approval_history
    import agent_graph
    import agent_tools
except ModuleNotFoundError as e:
    st.error(f"Import Error: {e}. Please ensure you are running this from the project root.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic Retail Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Strict Palette & Typography) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif !important;
}

/* Strict Palette: #FFFFFF and #4B164C */
.stApp, [data-testid="stAppViewContainer"], .main {
    background-color: #FFFFFF !important;
    color: #4B164C !important;
}

h1, h2, h3, h4, h5, h6 { 
    color: #4B164C !important; 
    font-weight: 700 !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #4B164C !important;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Metric Cards */
[data-testid="stMetricValue"] div {
    color: #4B164C !important; 
}
[data-testid="stMetricLabel"] p {
    color: #4B164C !important; 
    font-weight: 600 !important;
}

/* Clean UI / Remove separators */
hr {
    display: none;
}

.stAlert, .stInfo, .stSuccess, .stWarning {
    background-color: #FFFFFF !important;
    border: 2px solid #4B164C !important;
    color: #4B164C !important;
}

/* Buttons */
.stButton > button {
    background-color: #4B164C !important;
    color: #FFFFFF !important;
    border: 2px solid #4B164C !important;
    border-radius: 0px !important;
}

/* Chat Box */
.chat-box { 
    background-color: #FFFFFF; 
    padding: 20px; 
    border: 2px solid #4B164C; 
    margin-bottom: 20px; 
    color: #4B164C;
}
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data

def load_data():
    clean_df = pd.DataFrame()
    risk_df = pd.DataFrame()
    forecast_df = pd.DataFrame()
    
    if os.path.exists(CLEANED_DATA_PATH):
        clean_df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['date'])
        if 'revenue' not in clean_df.columns and not clean_df.empty:
            clean_df['revenue'] = clean_df['units_sold'] * clean_df['price']
            
    if os.path.exists(RISK_OUTPUT_PATH):
        risk_df = pd.read_csv(RISK_OUTPUT_PATH)
        
    if os.path.exists(FORECAST_OUTPUT_PATH):
        forecast_df = pd.read_csv(FORECAST_OUTPUT_PATH)
            
    return clean_df, risk_df, forecast_df

clean_df, risk_df, forecast_df = load_data()

# --- Helper for Plotly Styling ---
def apply_strict_theme(fig):
    fig.update_layout(
        plot_bgcolor="#FFFFFF", 
        paper_bgcolor="#FFFFFF", 
        font_color="#4B164C",
        margin=dict(t=30, b=10, l=10, r=10)
    )
    fig.update_traces(marker_color="#4B164C")
    return fig

# --- Sidebar Navigation ---
st.sidebar.title("Retail Copilot")
page = st.sidebar.radio("Navigation", [
    "1. Home", 
    "2. Data Quality",
    "3. Sales Dashboard", 
    "4. Inventory Dashboard", 
    "5. Forecast Dashboard",
    "6. Action Plan",
    "7. Agent Chat",
    "8. Agent Trace",
    "9. Governance"
])

if risk_df.empty or clean_df.empty:
    st.sidebar.warning("SYSTEM NOTICE: Output files missing. Run backend pipelines.")

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
    height: 250px !important;
}

/* Align the alert boxes content */
div[data-testid="stAlert"] > div {
    height: 100%;
}

</style>
""", unsafe_allow_html=True)

if page == "1. Home":
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
# PAGE 2: DATA Quality
# ==========================================
elif page == "2. Data Quality":

    st.title("Data Quality Assessment")
    st.write("Review dataset readiness for machine learning modeling.")

    if not clean_df.empty:

        # Backend report
        report = generate_quality_report(clean_df)
        shape_data = report.get("shape", {})

        # ===========================
        # Main Metrics
        # ===========================
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Rows",
            f"{shape_data.get('total_rows', len(clean_df)):,}"
        )

        col2.metric(
            "Total Columns",
            shape_data.get('total_columns', len(clean_df.columns))
        )

        col3.metric(
            "Missing Values",
            sum(report.get("missing_values", {}).values())
        )

        col4.metric(
            "Duplicate Count",
            report.get("duplicates", {}).get("exact_duplicate_rows", clean_df.duplicated().sum())
        )

        st.markdown("---")

        # ===========================
        # Dataset Summary
        # ===========================
        st.subheader("Dataset Summary")

        # Removed the Outlier Handling row here
        summary_df = pd.DataFrame({
            "Category": [
                "Date Range",
                "Number of Stores",
                "Number of Products"
            ],
            "Details": [
                f"{clean_df['date'].min().date()} → {clean_df['date'].max().date()}",
                f"{clean_df['store_id'].nunique()} Stores",
                f"{clean_df['product_id'].nunique()} Products"
            ]
        })

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")

        # ===========================
        # Charts
        # ===========================
        st.markdown("### Visual Distributions")
        
        # 1. Missing Values (Full Width)
        st.subheader("Missing Values")
        missing_df = clean_df.isna().sum().reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        
        # Guardrail: Only draw the bar chart if there are actually missing values to show
        if missing_df["Missing Count"].sum() == 0:
            st.success("✅ Dataset is perfectly clean. No missing values detected across any columns.")
        else:
            fig_miss = px.bar(
                missing_df,
                x="Column",
                y="Missing Count"
            )
            st.plotly_chart(apply_strict_theme(fig_miss), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True) # Spacer

        # 2. Histograms (Side-by-Side)
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Sales Distribution")
            fig_sales = px.histogram(
                clean_df,
                x="units_sold",
                nbins=40
            )
            st.plotly_chart(apply_strict_theme(fig_sales), use_container_width=True)

        with c2:
            st.subheader("Inventory Distribution")
            fig_inv = px.histogram(
                clean_df,
                x="inventory_level",
                nbins=40
            )
            st.plotly_chart(apply_strict_theme(fig_inv), use_container_width=True)

    else:
        st.warning("Cleaned data not available.")


# ==========================================
# PAGE 3: SALES DASHBOARD
# ==========================================
elif page == "3. Sales Dashboard":
    st.title("Sales Performance")
    
    if not clean_df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"${clean_df['revenue'].sum():,.0f}")
        col2.metric("Total Sales Units", f"{clean_df['units_sold'].sum():,.0f}")
        col3.metric("Average Selling Price", f"${clean_df['price'].mean():.2f}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sales Trend Over Time")
            daily_sales = clean_df.groupby('date')['revenue'].sum().reset_index()
            fig_trend = px.line(daily_sales, x='date', y='revenue')
            st.plotly_chart(apply_strict_theme(fig_trend), use_container_width=True)
            
        with c2:
            st.subheader("Top Products by Revenue")
            top_prod = clean_df.groupby('product_id')['revenue'].sum().nlargest(5).reset_index()
            fig_prod = px.bar(top_prod, x='product_id', y='revenue')
            st.plotly_chart(apply_strict_theme(fig_prod), use_container_width=True)

# ==========================================
# PAGE 4: INVENTORY DASHBOARD
# ==========================================
elif page == "4. Inventory Dashboard":
    st.title("Inventory Position & Risk")
    
    if not risk_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Current Inventory", f"{risk_df['inventory_level'].sum():,.0f}")
        
        safe_cover = risk_df['inventory_cover_days'].replace([np.inf, -np.inf], np.nan)
        col2.metric("Avg Inventory Cover", f"{safe_cover.mean():.1f} Days")
        col3.metric("Critical Risks", len(risk_df[risk_df['risk_bucket'] == 'Critical']))
        col4.metric("Overstock", len(risk_df[risk_df['risk_bucket'] == 'Overstock']))
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk Bucket Breakdown")
            fig_risk = px.pie(risk_df, names='risk_bucket', color_discrete_sequence=['#4B164C', '#FFFFFF'])
            fig_risk.update_layout(paper_bgcolor="#FFFFFF", font_color="#4B164C")
            fig_risk.update_traces(marker=dict(line=dict(color='#4B164C', width=2)))
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with c2:
            st.subheader("Product-Store Risk Table")
            st.dataframe(risk_df[['store_id', 'product_id', 'inventory_cover_days', 'risk_bucket']].sort_values('inventory_cover_days'), use_container_width=True)

# ==========================================
# PAGE 5: FORECAST DASHBOARD
# ==========================================
elif page == "5. Forecast Dashboard":
    st.title("Demand Forecast Accuracy")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast Period", "Next 7 Days")
    col2.metric("MAE", "59.74")
    col3.metric("WMAPE", "43.96%")
    
    if not forecast_df.empty:
        st.subheader("Forecast by Product (Next 7 Days)")
        forecast_prod = forecast_df.groupby('product_id')['forecast_7_day_demand'].sum().reset_index()
        fig_fcast = px.bar(forecast_prod, x='product_id', y='forecast_7_day_demand')
        st.plotly_chart(apply_strict_theme(fig_fcast), use_container_width=True)

# ==========================================
# PAGE 6: ACTION PLAN
# ==========================================
elif page == "6. Action Plan":
    st.title("Recommended Action Plan")
    st.write("Review simulated actions recommended by the Agent rules engine. Requires human approval before execution.")
    
    if not risk_df.empty:
        actionable_df = risk_df[risk_df['risk_bucket'].isin(['Critical', 'Warning', 'Overstock'])].copy()
        
        if not actionable_df.empty:
            actionable_df['recommended_action'] = np.select(
                [
                    actionable_df['risk_bucket'] == 'Critical',
                    actionable_df['risk_bucket'] == 'Warning',
                    actionable_df['risk_bucket'] == 'Overstock'
                ],
                ['Reorder urgently', 'Replenish / monitor', 'Review markdown'],
                default='None'
            )
            
            actionable_df['recommended_quantity'] = np.where(
                actionable_df['risk_bucket'].isin(['Critical', 'Warning']),
                np.maximum((actionable_df['forecast_7_day_demand'] * 1.5) - actionable_df['inventory_level'], 0).astype(int),
                0
            )
            
            display_df = actionable_df[['store_id', 'product_id', 'risk_bucket', 'recommended_action', 'recommended_quantity']].copy()
            display_df['approval_status'] = 'Pending'
            display_df['action_id'] = display_df['store_id'] + "_" + display_df['product_id']
            
            edited_df = st.data_editor(
                display_df, 
                column_config={
                    "approval_status": st.column_config.SelectboxColumn(
                        "Approval Status",
                        options=["Pending", "Approved", "Rejected"],
                        required=True,
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("Submit Decisions"):
                decisions_made = edited_df[edited_df['approval_status'] != 'Pending']
                for _, row in decisions_made.iterrows():
                    action_row = row.to_dict()
                    is_approved = row['approval_status'] == 'Approved'
                    decision_record = submit_approval_decision(action_row, approved=is_approved)
                    save_approved_action(decision_record)
                st.success(f"Processed {len(decisions_made)} decisions.")
                
            st.subheader("Approval History")
            st.dataframe(get_approval_history(), use_container_width=True)

# ==========================================
# PAGE 7: AGENT CHAT
# ==========================================
elif page == "7. Agent Chat":
    st.title("Agentic Retail Operations Copilot")
    
    st.markdown("### Ask a business question:")
    st.write("Example: Find the top 5 products at risk")
    
    query = st.text_input("", placeholder="Enter your query...")
    
    if st.button("Submit Query"):
        if query and "top" in query.lower() and "risk" in query.lower():
            st.markdown('<div class="chat-box">', unsafe_allow_html=True)
            st.markdown("**Agent Response (Live via agent_graph.py):**")
            
            try:
                # Trigger actual LangGraph Agent
                top_risk = agent_graph.find_top_risk_products(n=5, risk_data_path=RISK_OUTPUT_PATH)
                for item in top_risk:
                    st.write(f"**{item['product_id']} @ {item['store_id']} ({item['risk_bucket']}):**")
                    st.write(f"> {item['summary']}")
                    
                    if item['approval']['approval_status'] == 'pending':
                        c1, c2 = st.columns([1, 10])
                        with c1:
                            st.button(f"Approve {item['product_id']}", key=f"app_{item['product_id']}")
                        with c2:
                            st.button(f"Reject {item['product_id']}", key=f"rej_{item['product_id']}")
            except Exception as e:
                st.write(f"Agent execution failed: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        elif query:
            # Fallback for single item (e.g., "Why is P001 at S003 at risk?")
            try:
                st.markdown('<div class="chat-box">', unsafe_allow_html=True)
                result = agent_graph.retail_agent.invoke({"store_id": "S003", "product_id": "P001"})
                st.markdown("**Agent Response:**")
                st.write(result["summary"])
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.write("Could not parse specific store/product from query.")

# ==========================================
# PAGE 8: AGENT TRACE
# ==========================================
elif page == "8. Agent Trace":
    st.title("Agent Execution Trace")
    st.write("Live transparency log from `agent_tools.py`.")
    
    traces = agent_tools.get_trace_log()
    if traces:
        trace_df = pd.DataFrame(traces)
        st.dataframe(trace_df[['timestamp', 'step']], use_container_width=True)
        
        st.subheader("Raw Output Payload")
        st.json(traces[-1] if traces else {})
    else:
        st.info("No agent tasks have been executed in this session yet. Visit the Agent Chat to run a query.")

# ==========================================
# PAGE 9: GOVERNANCE
# ==========================================
elif page == "9. Governance":
    st.title("Responsible AI & Governance")
    
    with st.expander("Model Card (Random Forest Forecaster)", expanded=True):
        st.write("Algorithm: Random Forest Regressor\n\nObjective: Predict 7-day future demand.")
        
    with st.expander("Agent Card (LangGraph Supervisor)"):
        st.write("Role: Orchestrate business tool execution.\n\nGuardrails: Strictly forbidden from executing write-operations without `request_human_approval`.")
        
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import numpy as np
# import os
# import sys

# # --- Fix Python Path ---
# # This tells Python to look in the 'src' folder at the root of your project
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
# sys.path.append(os.path.join(root_dir, "src"))

# # Connect to actual project modules
# from data_quality import generate_quality_report
# from approval import submit_approval_decision, save_approved_action, get_approval_history
# from config import CLEANED_DATA_PATH, RISK_OUTPUT_PATH, FORECAST_OUTPUT_PATH

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Retail Operations Platform",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Custom High-Contrast UI ---
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&display=swap');

# html, body, [class*="css"], .stMarkdown, p, span, div {
#     font-family: 'Orbitron', sans-serif !important;
# }

# /* Base Palette Restrictions */
# .stApp, [data-testid="stAppViewContainer"], .main {
#     background-color: #FFFFFF !important;
#     color: #4B164C !important;
# }

# h1, h2, h3, h4, h5, h6 { 
#     color: #4B164C !important; 
#     font-weight: 700 !important;
# }

# /* Sidebar Styling */
# [data-testid="stSidebar"] {
#     background-color: #4B164C !important;
# }
# [data-testid="stSidebar"] * {
#     color: #FFFFFF !important;
# }

# /* Metric Cards */
# [data-testid="stMetricValue"] div {
#     color: #4B164C !important; 
# }
# [data-testid="stMetricLabel"] p {
#     color: #4B164C !important; 
#     font-weight: 600 !important;
# }

# /* Clean UI / Remove separators and dull elements */
# hr {
#     border-top: 2px solid #4B164C !important;
#     margin-top: 1.5em;
#     margin-bottom: 1.5em;
# }

# .stAlert {
#     background-color: #FFFFFF !important;
#     border: 1px solid #4B164C !important;
#     color: #4B164C !important;
# }

# /* Buttons */
# .stButton > button {
#     background-color: #4B164C !important;
#     color: #FFFFFF !important;
#     border: 2px solid #4B164C !important;
#     border-radius: 0px !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # --- Data Loading ---
# @st.cache_data
# def load_data():
#     clean_df = pd.DataFrame()
#     risk_df = pd.DataFrame()
#     forecast_df = pd.DataFrame()
    
#     # Load Clean Data
#     if os.path.exists(CLEANED_DATA_PATH):
#         clean_df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['date'])
#         if 'revenue' not in clean_df.columns and not clean_df.empty:
#             clean_df['revenue'] = clean_df['units_sold'] * clean_df['price']
            
#     # Load Risk Data 
#     if os.path.exists(RISK_OUTPUT_PATH):
#         risk_df = pd.read_csv(RISK_OUTPUT_PATH)

#     # Load Forecast Data
#     if os.path.exists(FORECAST_OUTPUT_PATH):
#         forecast_df = pd.read_csv(FORECAST_OUTPUT_PATH)
            
#     return clean_df, risk_df, forecast_df

# clean_df, risk_df, forecast_df = load_data()

# # --- Sidebar Navigation ---
# st.sidebar.title("Operations Platform")
# st.sidebar.markdown("<hr style='border-top: 1px solid #FFFFFF;'>", unsafe_allow_html=True)
# page = st.sidebar.radio("Navigation", [
#     "HOME", 
#     "DATA QUALITY",
#     "SALES DASHBOARD", 
#     "INVENTORY DASHBOARD", 
#     "FORECAST DASHBOARD",
#     "ACTION PLAN",
#     "GOVERNANCE"
# ])
# st.sidebar.markdown("<hr style='border-top: 1px solid #FFFFFF;'>", unsafe_allow_html=True)

# if risk_df.empty or clean_df.empty:
#     st.sidebar.warning("SYSTEM NOTICE: Output files missing. Run backend pipelines.")

# # ==========================================
# # PAGE 1: HOME
# # ==========================================
# if page == "HOME":
#     st.title("Retail Operations Platform")
    
#     st.markdown("### System Overview")
#     st.write("Deterministic retail operations monitor. Processes pipeline output for demand forecasting, stock-out detection, and automated business rule application.")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("### Active Pipelines")
#         st.info("""
#         * Ingestion & Feature Engineering
#         * XGBoost Demand Forecasting
#         * Deterministic Risk Assessment
#         * Human-in-the-loop Governance
#         """)

#     with col2:
#         st.markdown("### Status")
#         st.success(f"""
#         * Cleaned Data: {'ONLINE' if not clean_df.empty else 'OFFLINE'}
#         * Forecast Output: {'ONLINE' if not forecast_df.empty else 'OFFLINE'}
#         * Risk Matrix: {'ONLINE' if not risk_df.empty else 'OFFLINE'}
#         """)

# # ==========================================
# # PAGE 2: DATA QUALITY
# # ==========================================
# elif page == "DATA QUALITY":
#     st.title("Data Quality Assessment")
#     st.write("Real-time pipeline integrity check via `data_quality.py`.")

#     if not clean_df.empty:
#         # Connect directly to the Python function
#         report = generate_quality_report(clean_df)
#         shape_data = report.get("shape", {})
        
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("Total Rows", f"{shape_data.get('total_rows', 0):,}")
#         col2.metric("Total Columns", shape_data.get('total_columns', 0))
        
#         missing_count = sum(report.get("missing_values", {}).values())
#         col3.metric("Missing Values", missing_count)
        
#         dup_count = report.get("duplicates", {}).get("exact_duplicate_rows", 0)
#         col4.metric("Duplicate Count", dup_count)

#         st.markdown("---")
        
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             st.subheader("Sales Distribution")
#             fig_sales = px.histogram(clean_df, x="units_sold", nbins=40, color_discrete_sequence=['#4B164C'])
#             fig_sales.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color="#4B164C")
#             st.plotly_chart(fig_sales, use_container_width=True)

#         with c2:
#             st.subheader("Inventory Distribution")
#             fig_inv = px.histogram(clean_df, x="inventory_level", nbins=40, color_discrete_sequence=['#4B164C'])
#             fig_inv.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color="#4B164C")
#             st.plotly_chart(fig_inv, use_container_width=True)
            
#         with c3:
#             st.subheader("Anomaly Flags")
#             st.json({
#                 "Negative Values": report.get("negative_values", "None Detected"),
#                 "Date Continuity Gaps": report.get("date_continuity", {}).get("store_product_combos_with_gaps", 0)
#             })
#     else:
#         st.warning("Data not available. Run preprocessing pipeline.")

# # ==========================================
# # PAGE 3: SALES DASHBOARD
# # ==========================================
# elif page == "SALES DASHBOARD":
#     st.title("Sales Performance")
    
#     if not clean_df.empty:
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Total Revenue", f"${clean_df['revenue'].sum():,.0f}")
#         col2.metric("Total Sales Units", f"{clean_df['units_sold'].sum():,.0f}")
#         col3.metric("Average Selling Price", f"${clean_df['price'].mean():.2f}")
        
#         st.markdown("---")
        
#         c1, c2 = st.columns(2)
#         with c1:
#             st.subheader("Sales Trend Over Time")
#             daily_sales = clean_df.groupby('date')['revenue'].sum().reset_index()
#             fig_trend = px.line(daily_sales, x='date', y='revenue', color_discrete_sequence=['#4B164C'])
#             fig_trend.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color="#4B164C")
#             st.plotly_chart(fig_trend, use_container_width=True)
            
#         with c2:
#             st.subheader("Promotion Impact")
#             promo_sales = clean_df.groupby('holiday_or_promo_flag')['revenue'].sum().reset_index()
#             promo_sales['holiday_or_promo_flag'] = promo_sales['holiday_or_promo_flag'].map({0: 'Regular', 1: 'Promotion'})
#             fig_promo = px.pie(promo_sales, names='holiday_or_promo_flag', values='revenue', hole=0.4, color_discrete_sequence=['#4B164C', '#8A2B8A'])
#             fig_promo.update_layout(paper_bgcolor="#FFFFFF", font_color="#4B164C")
#             st.plotly_chart(fig_promo, use_container_width=True)

# # ==========================================
# # PAGE 4: INVENTORY DASHBOARD
# # ==========================================
# elif page == "INVENTORY DASHBOARD":
#     st.title("Inventory Position & Risk")
    
#     if not risk_df.empty:
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("Total Inventory", f"{risk_df['inventory_level'].sum():,.0f}")
        
#         # Guard against inf values for cover days
#         safe_cover = risk_df['inventory_cover_days'].replace([np.inf, -np.inf], np.nan)
#         col2.metric("Avg Cover", f"{safe_cover.mean():.1f} Days")
        
#         col3.metric("Critical Risks", len(risk_df[risk_df['risk_bucket'] == 'Critical']))
#         col4.metric("Overstock", len(risk_df[risk_df['risk_bucket'] == 'Overstock']))
        
#         st.markdown("---")
        
#         c1, c2 = st.columns(2)
#         with c1:
#             st.subheader("Risk Bucket Breakdown")
#             # Ensure high contrast palette is respected
#             fig_risk = px.pie(risk_df, names='risk_bucket', color_discrete_sequence=['#4B164C', '#6E2070', '#8A2B8A', '#A33AA3'])
#             fig_risk.update_layout(paper_bgcolor="#FFFFFF", font_color="#4B164C")
#             st.plotly_chart(fig_risk, use_container_width=True)
            
#         with c2:
#             st.subheader("Product-Store Risk Matrix")
#             st.dataframe(risk_df[['store_id', 'product_id', 'inventory_cover_days', 'risk_bucket']].sort_values('inventory_cover_days'), use_container_width=True)

# # ==========================================
# # PAGE 5: FORECAST DASHBOARD
# # ==========================================
# elif page == "FORECAST DASHBOARD":
#     st.title("Demand Forecast Projections")
    
#     if not forecast_df.empty:
#         st.write(f"Total Model Projections: **{len(forecast_df)}**")
        
#         c1, c2 = st.columns(2)
#         with c1:
#             st.subheader("Forecast by Product (Next 7 Days)")
#             forecast_prod = forecast_df.groupby('product_id')['forecast_7_day_demand'].sum().reset_index()
#             fig_fcast = px.bar(forecast_prod, x='product_id', y='forecast_7_day_demand', color_discrete_sequence=['#4B164C'])
#             fig_fcast.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color="#4B164C")
#             st.plotly_chart(fig_fcast, use_container_width=True)
            
#         with c2:
#             st.subheader("Forecast Raw Output Table")
#             st.dataframe(forecast_df, use_container_width=True)
#     else:
#         st.warning("Forecast data missing. Run forecasting pipeline.")

# # ==========================================
# # PAGE 6: ACTION PLAN
# # ==========================================
# elif page == "ACTION PLAN":
#     st.title("Recommended Action Plan")
#     st.write("Review simulated actions. Submitting decisions writes directly to system records via `approval.py`.")
    
#     if not risk_df.empty:
#         # Isolate items needing action
#         actionable_df = risk_df[risk_df['risk_bucket'].isin(['Critical', 'Warning', 'Overstock'])].copy()
        
#         if not actionable_df.empty:
#             actionable_df['recommended_action'] = np.select(
#                 [
#                     actionable_df['risk_bucket'] == 'Critical',
#                     actionable_df['risk_bucket'] == 'Warning',
#                     actionable_df['risk_bucket'] == 'Overstock'
#                 ],
#                 ['Reorder urgently', 'Replenish / monitor', 'Review markdown'],
#                 default='None'
#             )
            
#             actionable_df['recommended_quantity'] = np.where(
#                 actionable_df['risk_bucket'].isin(['Critical', 'Warning']),
#                 np.maximum((actionable_df['forecast_7_day_demand'] * 1.5) - actionable_df['inventory_level'], 0).astype(int),
#                 0
#             )
            
#             # Setup interactive dataframe
#             display_df = actionable_df[['store_id', 'product_id', 'risk_bucket', 'recommended_action', 'recommended_quantity']].copy()
#             display_df['approval_status'] = 'Pending'
#             display_df['action_id'] = display_df['store_id'] + "_" + display_df['product_id']
            
#             st.write("Modify the `approval_status` column and click Save to execute.")
            
#             edited_df = st.data_editor(
#                 display_df, 
#                 column_config={
#                     "approval_status": st.column_config.SelectboxColumn(
#                         "Approval Status",
#                         options=["Pending", "Approved", "Rejected"],
#                         required=True,
#                     )
#                 },
#                 use_container_width=True,
#                 hide_index=True,
#                 key="action_editor"
#             )
            
#             if st.button("Save Selected Decisions"):
#                 decisions_made = edited_df[edited_df['approval_status'] != 'Pending']
#                 for _, row in decisions_made.iterrows():
#                     action_row = row.to_dict()
#                     is_approved = row['approval_status'] == 'Approved'
                    
#                     # Interface with approval.py
#                     decision_record = submit_approval_decision(action_row, approved=is_approved)
#                     save_approved_action(decision_record)
                    
#                 st.success(f"Successfully processed {len(decisions_made)} decisions.")
                
#             st.markdown("---")
#             st.subheader("Historical Log")
#             history_df = get_approval_history()
#             st.dataframe(history_df, use_container_width=True)
            
#         else:
#             st.info("No actionable risks currently flagged in the system.")

# # ==========================================
# # PAGE 7: GOVERNANCE
# # ==========================================
# elif page == "GOVERNANCE":
#     st.title("Governance & Control")
    
#     with st.expander("Model Card (XGBoost Forecaster)", expanded=True):
#         st.write("""
#         * **Algorithm:** XGBoost Regressor
#         * **Objective:** Predict 7-day future demand for product-store combinations.
#         * **Limitations:** Assumes historical lag patterns will continue.
#         """)
        
#     with st.expander("Business Rules & Human Approval Policy"):
#         st.write("""
#         * **Approval Policy:** All simulated actions generated by the Recommendation Tool MUST be reviewed and manually signed off.
#         * Execution is logged chronologically via `approval.py`.
#         """)