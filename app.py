import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==============================
# ✅ Safe Model Loading Section
# ==============================
@st.cache_resource
def load_model(file_path):
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        st.stop()
    except ModuleNotFoundError as e:
        st.error("⚠️ Model environment mismatch or missing dependency.")
        st.write(f"Missing module: {e.name}")
        st.info("Try retraining or re-saving your model using joblib in the same environment.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading {file_path}")
        st.exception(e)
        st.stop()

# Load model and columns safely
model = load_model('trained_model_joblib.pkl')
model_columns = load_model('model_columns.pkl')

if model is None or model_columns is None:
    st.stop()

# ==============================
# 🎨 App UI Starts Here
# ==============================
st.title('🛍️ E-commerce Sales Prediction App')
st.markdown("Enter the features below to predict the sales amount.")

st.header("🧮 Enter Feature Values")

# Create input fields dynamically
input_data = {}

# Numerical features
for i, col in enumerate(model_columns):
    if col in ['Quantity', 'Unit Price', 'Discount', 'Profit', 'Year', 'Month', 'Day', 'Quarter',
               'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'Product_Avg_Sales',
               'Product_Avg_Profit', 'Product_Order_Count']:
        if col in ['Discount', 'Year', 'Month', 'Day', 'Quarter']:
            input_data[col] = st.number_input(f'Enter {col}', step=1, value=0, key=f"{col}_num")
        elif col == 'Quantity':
            input_data[col] = st.number_input(f'Enter {col}', min_value=1, step=1, value=1, key=f"{col}_num")
        else:
            input_data[col] = st.number_input(f'Enter {col}', value=0.0, key=f"{col}_num")

# Region
regions = ['North', 'East', 'South', 'West']
selected_region = st.selectbox('Select Region', regions)
for region in regions:
    input_data[f'Region_{region}'] = (selected_region == region)

# City
cities = ['Bangalore', 'Delhi', 'Patna', 'Kolkata', 'Pune', 'Mumbai', 'Chennai', 'Hyderabad',
           'Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Visakhapatnam',
           'Bhopal', 'Pimpri-Chinchwad', 'Agra', 'Nashik', 'Faridabad']
selected_city = st.selectbox('Select City', cities)
for city in cities:
    input_data[f'City_{city}'] = (selected_city == city)

# Category
categories = ['Books', 'Groceries', 'Kitchen', 'Clothing', 'Electronics',
              'Furniture', 'Toys', 'Sports', 'Footwear', 'Beauty']
selected_category = st.selectbox('Select Category', categories)
for category in categories:
    input_data[f'Category_{category}'] = (selected_category == category)

# Sub-Category
sub_categories = [col.replace('Sub-Category_', '') for col in model_columns if col.startswith('Sub-Category_')]
if len(sub_categories) > 0:
    selected_sub_category = st.selectbox('Select Sub-Category', sub_categories)
    for sub_category in sub_categories:
        input_data[f'Sub-Category_{sub_category}'] = (selected_sub_category == sub_category)

# Payment Mode
payment_modes = ['Credit Card', 'Debit Card', 'UPI', 'Cash On Delivery', 'Other']
selected_payment_mode = st.selectbox('Select Payment Mode', payment_modes)
for payment_mode in payment_modes:
    input_data[f'Payment Mode_{payment_mode}'] = (selected_payment_mode == payment_mode)

# Weekday
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_weekday = st.selectbox('Select Weekday', weekdays)
for weekday in weekdays:
    input_data[f'Weekday_{weekday}'] = (selected_weekday == weekday)

# Weekend
selected_is_weekend = st.selectbox('Is Weekend?', ['False', 'True'])
input_data['Is_Weekend_True'] = (selected_is_weekend == 'True')

# Ensure all model columns exist
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

# ==============================
# 🔮 Prediction Section
# ==============================
if st.button('🔍 Predict Sales'):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[model_columns]

        # Convert boolean to int
        bool_cols_input = input_df.select_dtypes(include=['bool']).columns
        input_df[bool_cols_input] = input_df[bool_cols_input].astype(int)

        prediction = model.predict(input_df)
        st.metric("📈 Predicted Sales", f"{prediction[0]:,.2f}")
    except Exception as e:
        st.error("⚠️ Prediction failed.")
        st.exception(e)
