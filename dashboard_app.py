import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="AI-Driven E-commerce Dashboard", layout="wide")

# ============ Load Data ============
@st.cache_data
def load_data():
    return pd.read_csv("data/Ecommerce_Sales_Data_2024_2025.csv")

@st.cache_resource
def load_model():
    with open("model/trained_model.pkl", "rb") as f:
        return pickle.load(f)

data = load_data()
model = load_model()

st.title("🧠 AI-Driven E-commerce Sales Dashboard")
st.markdown("Analyze your sales performance and predict future sales with AI.")

# ============ Sidebar Filters ============
st.sidebar.header("🔍 Filters")

year = st.sidebar.selectbox("Select Year", sorted(data["Year"].unique()))
region = st.sidebar.multiselect("Select Region", data["Region"].unique(), default=data["Region"].unique())
category = st.sidebar.multiselect("Select Category", data["Category"].unique(), default=data["Category"].unique())

filtered_data = data[(data["Year"] == year) & 
                     (data["Region"].isin(region)) &
                     (data["Category"].isin(category))]

# ============ KPI Section ============
st.subheader(f"📊 Summary for {year}")
total_sales = filtered_data["Sales"].sum()
avg_profit = filtered_data["Profit"].mean()
total_orders = filtered_data["Order ID"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Average Profit", f"${avg_profit:,.2f}")
col3.metric("Total Orders", f"{total_orders:,}")

# ============ Charts ============
tab1, tab2, tab3 = st.tabs(["📈 Sales Trends", "🏆 Top Products", "📍 Regional Insights"])

with tab1:
    trend_chart = px.line(filtered_data, x="Month", y="Sales", color="Category", title="Monthly Sales Trend")
    st.plotly_chart(trend_chart, use_container_width=True)

with tab2:
    top_products = filtered_data.groupby("Product Name")["Sales"].sum().nlargest(10).reset_index()
    bar_chart = px.bar(top_products, x="Product Name", y="Sales", title="Top 10 Products by Sales")
    st.plotly_chart(bar_chart, use_container_width=True)

with tab3:
    region_chart = px.pie(filtered_data, values="Sales", names="Region", title="Sales by Region")
    st.plotly_chart(region_chart, use_container_width=True)

# ============ AI Prediction Section ============
st.subheader("🤖 Predict Future Sales")

avg_input = filtered_data[["Quantity", "Unit Price", "Discount", "Profit"]].mean().to_dict()
input_df = pd.DataFrame([avg_input])
prediction = model.predict(input_df)[0]

st.success(f"Predicted Next Month Sales: **${prediction:,.2f}**")

st.caption("Model-driven prediction using historical averages for selected filters.")
