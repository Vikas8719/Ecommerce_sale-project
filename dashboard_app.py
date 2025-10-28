import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="🧠 AI-Driven E-commerce Dashboard", layout="wide")

# ============ Load Data ============
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_ecommerce_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("trained_model_joblib.pkl")


data = load_data()
model = load_model()

st.title("🧠 AI-Driven E-commerce Sales Dashboard")
st.markdown("Analyze your sales performance and predict future sales with AI insights.")

# ============ Sidebar Filters ============
st.sidebar.header("🔍 Filters")

# Year Filter (if present)
if "Year" in data.columns:
    year = st.sidebar.selectbox("Select Year", sorted(data["Year"].unique()))
    data = data[data["Year"] == year]

# Region Filter (since it's encoded)
region_columns = [col for col in data.columns if col.startswith("Region_")]
region_names = [col.replace("Region_", "") for col in region_columns]
selected_regions = st.sidebar.multiselect("Select Region", region_names, default=region_names)

# Filter data based on selected regions
region_mask = data[region_columns].apply(lambda row: any(row[f"Region_{r}"] == 1 for r in selected_regions), axis=1)
filtered_data = data[region_mask]

# Category Filter (handle encoded columns)
category_columns = [col for col in data.columns if col.startswith("Category_")]
if category_columns:
    category_names = [col.replace("Category_", "") for col in category_columns]
    selected_categories = st.sidebar.multiselect("Select Category", category_names, default=category_names)
    category_mask = filtered_data[category_columns].apply(lambda row: any(row[f"Category_{c}"] == 1 for c in selected_categories), axis=1)
    filtered_data = filtered_data[category_mask]

# ============ KPI Section ============
st.subheader("📊 Summary Statistics")
if "Sales" in filtered_data.columns and "Profit" in filtered_data.columns:
    total_sales = filtered_data["Sales"].sum()
    avg_profit = filtered_data["Profit"].mean() if not filtered_data.empty else 0
    total_orders = filtered_data.shape[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Average Profit", f"${avg_profit:,.2f}")
    col3.metric("Total Records", f"{total_orders:,}")

# ============ Charts ============
tab1, tab2, tab3 = st.tabs(["📈 Sales Trends", "🏆 Top Products", "📍 Regional Insights"])

with tab1:
    if "Month" in filtered_data.columns:
        trend_chart = px.line(filtered_data, x="Month", y="Sales", title="Monthly Sales Trend")
        st.plotly_chart(trend_chart, use_container_width=True)
    else:
        st.info("Month column not found for trend analysis.")

with tab2:
    if "Product Name" in filtered_data.columns:
        top_products = filtered_data.groupby("Product Name")["Sales"].sum().nlargest(10).reset_index()
        bar_chart = px.bar(top_products, x="Product Name", y="Sales", title="Top 10 Products by Sales")
        st.plotly_chart(bar_chart, use_container_width=True)
    else:
        st.info("Product Name column not found for top products chart.")

with tab3:
    # Create a readable Region column for pie chart
    def get_region(row):
        for col in region_columns:
            if row[col] == 1:
                return col.replace("Region_", "")
        return "Other"

    filtered_data["Region_Name"] = filtered_data.apply(get_region, axis=1)
    region_chart = px.pie(filtered_data, values="Sales", names="Region_Name", title="Sales by Region")
    st.plotly_chart(region_chart, use_container_width=True)

# ============ AI Prediction Section ============
st.subheader("🤖 Predict Future Sales")

try:
    if all(col in filtered_data.columns for col in ["Quantity", "Unit Price", "Discount", "Profit"]):
        # Prepare full feature set for prediction
        feature_cols = model.feature_names_ if hasattr(model, "feature_names_") else data.columns.drop("Sales", errors="ignore")

        # Create average input row from filtered data (use numeric + encoded columns)
        avg_input = filtered_data[feature_cols].mean(numeric_only=True).to_dict()
        input_df = pd.DataFrame([avg_input])

        # Fill any missing model columns with 0 (important for one-hot encoded features)
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match model input order
        input_df = input_df[feature_cols]

        # Predict sales
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Next Month Sales: **${prediction:,.2f}**")
    else:
        st.warning("Prediction unavailable: Required columns missing in dataset.")

    st.caption("Model-driven prediction using historical averages for selected filters.")

except Exception as e:
    st.error("⚠️ Prediction failed. Please check model-data compatibility.")
    st.exception(e)
