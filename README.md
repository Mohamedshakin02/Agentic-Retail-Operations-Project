# Agentic-Retail-Operations-Project--Al-Futtaim
https://www.kaggle.com/code/beshoyemad24/retail-store-inventory-forecastin-eda-prediction#Exploratory-Data-Analysis-(EDA)

# Function: What it does
Function:What it does
add_date_features: Pulls day_of_week, week_of_year, month, is_weekend out of the date
add_lag_features: Adds "what were sales yesterday / 7 days ago for this exact product"
add_rolling_features: Adds 7/14/28-day rolling average sales, plus rolling inventory
add_price_features: Adds % price change vs. the day before
add_inventory_cover_feature: Adds the days-of-cover number
add_categorical_encoding: Turns text like "Electronics" into numbers models can use
build_features:Runs all of the above, in the right order, with a column-check first


# Inventory_risk
calculate_inventory_cover: The core math: inventory ÷ daily sales = days of cover
get_thresholds_for_category: Looks up category-specific Critical/Warning/Normal cutoffs
detect_stockout_risk: Classifies into Critical/Warning/Normal/Over
stockmerge_features_and_forecast: Joins the feature table with the forecast numbers
add_risk_columns: Runs the risk math across every row at once

# recommendation
recommend_business_action: Risk bucket → plain-English action
recommend_reorder_quantity: How many units to reorder
add_recommendation_columns / build_action_recommendation_table: Batch versions, produce the final CSV
get_top_risk_products: Answers "find the top N at risk" directly

# agent_tools
get_sales_summary, get_inventory_summary, forecast_demand, analyze_promotion_impact: Data lookups (still stubbed with fake data)
generate_business_summary: Turns the numbers into one readable sentence
log_agent_trace / get_trace_log: Records every step the agent takes

# agent_graph
AgentState: The shape of the "box" moving through the pipeline
node_get_inventory → node_request_approval: One station each, in order
retail_agent: The compiled, runnable agent

# approval - acts as a safety gate
request_human_approval: Marks something pending
submit_approval_decision: Records a yes/no
save_approved_action: Appends it to approved_actions.csv
get_approval_history: Reads it back for the UI

# rag - answers questions about our own documentation 
load_documents: Reads your .md files, splits into chunks
build_index: Converts each chunk into a vector (the embedding step)
query_docs: Finds the closest-matching chunks to a question
answer_doc_question: The main entry point, combining the above


# Tool Name: Purpose
get_sales_summary: Sales metrics
get_inventory_summary: Inventory + avg sales
forecast_demand: Predict demand
calculate_inventory_cover: Compute stock duration
detect_stockout_risk: Classify risk
analyze_promotion_impact: Promo analysis
recommend_business_action: Suggest action
recommend_reorder_quantity: Quantity to reorder
generate_business_summary: Explain result
request_human_approval: Approval checkpoint
log_agent_trace: Track agent reasoning
Column names list of cleaned data-

['date',
 'store_id',
 'product_id',
 'category',
 'region',
 'inventory_level',
 'units_sold',
 'units_ordered',
 'dataset_demand_forecast',
 'price',
 'discount',
 'weather_condition',
 'holiday_or_promo_flag',
 'competitor_price',
 'seasonality',
 'store_id_encoded',
 'product_id_encoded',
 'category_encoded',
 'region_encoded',
 'weather_condition_encoded',
 'seasonality_encoded',
 'units_sold_log',
 'demand_forecast_log']
