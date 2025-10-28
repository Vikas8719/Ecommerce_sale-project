import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ==============================
# ✅ Safe Model Loading Section
# ==============================
@st.cache_resource
def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        st.error("Model file was created in a different environment.")
        st.info("Try retraining or re-saving your model using joblib.")
        return None
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {file_path}")
        st.exception(e)
        st.stop()

# Load the trained model and columns safely
model = load_pickle_file('trained_model.pkl')
model_columns = load_pickle_file('model_columns.pkl')

if model is None or model_columns is None:
    st.stop()

# ==============================
# App UI Starts Here
# ==============================
st.title('Sales Prediction App')
st.write("Enter the features below to predict the sales.")

# Create input fields for features
input_data = {}
for i, col in enumerate(model_columns):
    if col in ['Quantity', 'Unit Price', 'Discount', 'Profit', 'Year', 'Month', 'Day', 'Quarter', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'Product_Avg_Sales', 'Product_Avg_Profit', 'Product_Order_Count']:
        # Numerical input
        if col in ['Discount', 'Year', 'Month', 'Day', 'Quarter']:
            input_data[col] = st.number_input(f'Enter {col}', step=1, value=0, key=f"{col}_num")
        elif col == 'Quantity':
            input_data[col] = st.number_input(f'Enter {col}', min_value=1, step=1, value=1, key=f"{col}_num")
        else:
            input_data[col] = st.number_input(f'Enter {col}', value=0.0, key=f"{col}_num")

    elif col.startswith('Region_'):
        region = col.replace('Region_', '')
        input_data[col] = st.selectbox(
            f'Select Region ({col})',
            ['North', 'East', 'South', 'West'],
            key=f"region_{col}"
        ) == region

    elif col.startswith('City_'):
        input_data[col] = False

    elif col.startswith('Category_'):
        input_data[col] = False

    elif col.startswith('Sub-Category_'):
        input_data[col] = False

    elif col.startswith('Payment Mode_'):
        payment_mode = col.replace('Payment Mode_', '')
        input_data[col] = st.selectbox(
            f'Select Payment Mode ({col})',
            ['Credit Card', 'Debit Card', 'UPI', 'Cash On Delivery', 'Other'],
            key=f"paymode_{col}"
        ) == payment_mode

    elif col.startswith('Weekday_'):
        input_data[col] = False

    elif col.startswith('Is_Weekend_'):
        input_data[col] = st.selectbox(
            f'Is Weekend? ({col})',
            ['False', 'True'],
            key=f"weekend_{col}"
        ) == 'True'


# Reconstruct original categorical features for selectboxes
regions = ['North', 'East', 'South', 'West']
selected_region = st.selectbox('Select Region', regions, key='main_region')
for region in regions:
    input_data[f'Region_{region}'] = (selected_region == region)

cities = ['Bangalore', 'Delhi', 'Patna', 'Kolkata', 'Pune', 'Mumbai', 'Chennai', 'Hyderabad', 'Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Visakhapatnam', 'Bhopal', 'Pimpri-Chinchwad', 'Agra', 'Nashik', 'Faridabad']
selected_city = st.selectbox('Select City', cities, key='main_city')
for city in cities:
    if f'City_{city}' in model_columns:
        input_data[f'City_{city}'] = (selected_city == city)
    else:
        input_data[f'City_{city}'] = False

categories = ['Books', 'Groceries', 'Kitchen', 'Clothing', 'Electronics', 'Furniture', 'Toys', 'Sports', 'Footwear', 'Beauty']
selected_category = st.selectbox('Select Category', categories, key='main_category')
for category in categories:
    if f'Category_{category}' in model_columns:
        input_data[f'Category_{category}'] = (selected_category == category)
    else:
        input_data[f'Category_{category}'] = False

sub_categories = [col.replace('Sub-Category_', '') for col in model_columns if col.startswith('Sub-Category_')]
selected_sub_category = st.selectbox('Select Sub-Category', sub_categories, key='main_sub_category')
for sub_category in sub_categories:
    if f'Sub-Category_{sub_category}' in model_columns:
        input_data[f'Sub-Category_{sub_category}'] = (selected_sub_category == sub_category)
    else:
        input_data[f'Sub-Category_{sub_category}'] = False

payment_modes = ['Credit Card', 'Debit Card', 'UPI', 'Cash On Delivery', 'Other']
selected_payment_mode = st.selectbox('Select Payment Mode', payment_modes, key='main_payment_mode')
for payment_mode in payment_modes:
    if f'Payment Mode_{payment_mode}' in model_columns:
        input_data[f'Payment Mode_{payment_mode}'] = (selected_payment_mode == payment_mode)
    else:
        input_data[f'Payment Mode_{payment_mode}'] = False

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_weekday = st.selectbox('Select Weekday', weekdays, key='main_weekday')
for weekday in weekdays:
    if f'Weekday_{weekday}' in model_columns:
        input_data[f'Weekday_{weekday}'] = (selected_weekday == weekday)
    else:
        input_data[f'Weekday_{weekday}'] = False

selected_is_weekend = st.selectbox('Is Weekend?', ['False', 'True'], key='main_is_weekend')
input_data['Is_Weekend_True'] = (selected_is_weekend == 'True')

# Ensure all columns are present in input_data
for col in model_columns:
    if col not in input_data:
        input_data[col] = False

if st.button('Predict Sales'):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model_columns]
    bool_cols_input = input_df.select_dtypes(include=['bool']).columns
    input_df[bool_cols_input] = input_df[bool_cols_input].astype(int)

    prediction = model.predict(input_df)
    st.success(f'Predicted Sales: {prediction[0]:,.2f}')
