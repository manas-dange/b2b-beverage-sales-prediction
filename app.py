import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# --- 1. Page Configuration (Must be first) ---
st.set_page_config(page_title="B2B Beverage Dashboard", layout="wide")

# --- 2. Load Models and Historical Data ---
@st.cache_resource
def load_tools():
    model = pickle.load(open('logistic_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model_columns = pickle.load(open('model_columns.pkl', 'rb'))
    return model, scaler, model_columns

@st.cache_data
def load_data():
    # Load Line Graph Data
    trend_df = pd.read_csv('monthly_trend.csv')
    trend_df['Order_Date'] = pd.to_datetime(trend_df['Order_Date'])
    trend_df.set_index('Order_Date', inplace=True)
    
    # Load Pie Chart Data
    category_df = pd.read_csv('category_share.csv')
    
    # Load Scatter Plot Data
    cluster_df = pd.read_csv('cluster_sample.csv')
    cluster_df['Cluster'] = cluster_df['Cluster'].astype(str) # For discrete colors
    
    return trend_df, category_df, cluster_df

model, scaler, model_columns = load_tools()
trend_df, category_df, cluster_df = load_data()

# --- 3. Header ---
st.title(" B2B Beverage Sales & Analytics Command Center")
st.markdown("Predict incoming B2B order values and analyze historical sales trends across 9 million transactions.")
st.divider()

# --- 4. TOP SECTION: PREDICTION ENGINE ---
st.header(" 1. Order Value Predictor")
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader(" Enter Order Details")
    with st.form("prediction_form"):
        quantity = st.number_input("Quantity Ordered", min_value=1, value=100)
        discount = st.number_input("Discount Applied (e.g., 0.05 for 5%)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        
        category = st.selectbox("Beverage Category", ["Alcoholic Beverages", "Soft Drinks", "Water", "Juices"])
        region = st.selectbox("Region", ["North", "South", "East", "West"]) 
        
        submit_button = st.form_submit_button(label="Analyze Order")

with col2:
    st.subheader(" AI Prediction Analytics")
    if submit_button:
        # Data Prep
        input_data = pd.DataFrame({'Quantity': [quantity], 'Discount': [discount], 'Category': [category], 'Region': [region]})
        input_encoded = pd.get_dummies(input_data)
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_aligned)
        
        # Prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        high_value_prob = probability[1] * 100
        standard_prob = probability[0] * 100
        
        # Metrics
        st.metric(
            label="Predicted AI Status", 
            value="HIGH-VALUE 💰" if prediction == 1 else "STANDARD ",
            delta=f"Confidence: {max(high_value_prob, standard_prob):.1f}%"
        )
        
        st.write("**Model Probability Breakdown:**")
        chart_data = pd.DataFrame({"Order Type": ["Standard", "High-Value"], "Probability (%)": [standard_prob, high_value_prob]}).set_index("Order Type")
        st.bar_chart(chart_data, color=["#FF4B4B"]) 
    else:
        st.info(" Enter details on the left and click 'Analyze Order'.")

st.divider()

# --- 5. BOTTOM SECTION: HISTORICAL INSIGHTS ---
st.header(" 2. Historical Business Insights")

# Row 1 of Charts: Line Graph and Pie Chart side-by-side
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader(" Monthly Revenue Trend")
    st.markdown("Identifies seasonal demand peaks for inventory planning.")
    st.line_chart(trend_df['Total_Price'], color="#1f77b4")

with chart_col2:
    st.subheader(" Revenue by Category")
    st.markdown("Shows product mix dominance.")
    fig_pie = px.pie(category_df, values='Total_Price', names='Category', hole=0.4, 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

# Row 2 of Charts: K-Means Scatter Plot spanning full width
st.subheader("🎯 Customer Segmentation (K-Means)")
st.markdown("B2B clients clustered by Revenue, Volume, and Average Discount received.")
fig_scatter = px.scatter(
    cluster_df, x='Total_Quantity', y='Total_Revenue', color='Cluster',
    size='Average_Discount', hover_data=['Customer_ID'],
    color_discrete_sequence=px.colors.qualitative.Vivid
)
st.plotly_chart(fig_scatter, use_container_width=True)