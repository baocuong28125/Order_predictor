import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Setting up the Streamlit app title
st.title("Complete Data Analysis and Modeling for Orders Sample")

# --- Step 1: Load and Preprocess Data ---
st.header("Step 1: Data Loading and Preprocessing")

@st.cache_data
def load_data():
    data = pd.read_csv("orders_sample_with_stock.csv")
    return data

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    missing_values = data.isnull().sum()
    data['Quantity_Ordered'] = pd.to_numeric(data['Quantity_Ordered'], errors='coerce')
    data['Unit_Price'] = pd.to_numeric(data['Unit_Price'], errors='coerce')
    data['Stock_Remaining'] = pd.to_numeric(data['Stock_Remaining'], errors='coerce')
    data = data.dropna()
    data['Total_Price'] = data['Quantity_Ordered'] * data['Unit_Price']
    return data, missing_values

st.subheader("Code for Loading and Preprocessing Data")
st.code("""
import pandas as pd
import io

@st.cache_data
def load_data():
    data = pd.read_csv("orders_sample_with_stock.csv")
    return data

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    missing_values = data.isnull().sum()
    data['Quantity_Ordered'] = pd.to_numeric(data['Quantity_Ordered'], errors='coerce')
    data['Unit_Price'] = pd.to_numeric(data['Unit_Price'], errors='coerce')
    data['Stock_Remaining'] = pd.to_numeric(data['Stock_Remaining'], errors='coerce')
    data = data.dropna()
    data['Total_Price'] = data['Quantity_Ordered'] * data['Unit_Price']
    return data, missing_values
""", language="python")

# Load and preprocess data
data = load_data()
processed_data, missing_values = preprocess_data(data)

# --- Step 2: Describe the Data ---
st.header("Step 2: Data Description")
st.write("This dataset contains order information for various Apple products, including iPhone 13, iPad Air, Apple Watch, MacBook Air, and AirPods Pro. The key columns are:")
st.write("- **Date**: Date of the order (converted to datetime).")
st.write("- **SKU**: Unique product identifier.")
st.write("- **Product_Name**: Name of the product (e.g., iPhone 13, AirPods Pro).")
st.write("- **Quantity_Ordered**: Number of units ordered.")
st.write("- **Unit_Price**: Price per unit in USD.")
st.write("- **Stock_Remaining**: Stock remaining after the order.")
st.write("- **Total_Price**: Calculated as Quantity_Ordered * Unit_Price.")

st.subheader("Basic Statistics")
st.write(processed_data.describe())
st.subheader("Missing Values")
st.write(missing_values)
st.subheader("Data Types")
st.write(processed_data.dtypes)

st.subheader("Code for Data Description")
st.code("""
st.header("Step 2: Data Description")
st.write("This dataset contains order information for various Apple products...")
st.write("- **Date**: Date of the order (converted to datetime).")
st.write("- **SKU**: Unique product identifier.")
st.write("- **Product_Name**: Name of the product (e.g., iPhone 13, AirPods Pro).")
st.write("- **Quantity_Ordered**: Number of units ordered.")
st.write("- **Unit_Price**: Price per unit in USD.")
st.write("- **Stock_Remaining**: Stock remaining after the order.")
st.write("- **Total_Price**: Calculated as Quantity_Ordered * Unit_Price.")
st.subheader("Basic Statistics")
st.write(processed_data.describe())
st.subheader("Missing Values")
st.write(missing_values)
st.subheader("Data Types")
st.write(processed_data.dtypes)
""", language="python")

# --- Step 3: Analyze and Visualize Data ---
st.header("Step 3: Data Analysis and Visualization")

# Chart 1: Total Quantity Ordered by Product
st.subheader("Chart 1: Total Quantity Ordered by Product")
qty_by_product = processed_data.groupby('Product_Name')['Quantity_Ordered'].sum().reset_index()
fig1 = px.bar(qty_by_product, x='Product_Name', y='Quantity_Ordered', title='Total Quantity Ordered by Product',
              color='Product_Name', color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig1)

st.subheader("Code for Chart 1")
st.code("""
qty_by_product = processed_data.groupby('Product_Name')['Quantity_Ordered'].sum().reset_index()
fig1 = px.bar(qty_by_product, x='Product_Name', y='Quantity_Ordered', title='Total Quantity Ordered by Product',
              color='Product_Name', color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig1)
""", language="python")

# Chart 2: Total Revenue by Product
st.subheader("Chart 2: Total Revenue by Product")
revenue_by_product = processed_data.groupby('Product_Name')['Total_Price'].sum().reset_index()
fig2 = px.pie(revenue_by_product, names='Product_Name', values='Total_Price', title='Total Revenue by Product',
              color_discrete_sequence=px.colors.qualitative.Set3)
st.plotly_chart(fig2)

st.subheader("Code for Chart 2")
st.code("""
revenue_by_product = processed_data.groupby('Product_Name')['Total_Price'].sum().reset_index()
fig2 = px.pie(revenue_by_product, names='Product_Name', values='Total_Price', title='Total Revenue by Product',
              color_discrete_sequence=px.colors.qualitative.Set3)
st.plotly_chart(fig2)
""", language="python")

# Chart 3: Orders Over Time
st.subheader("Chart 3: Orders Over Time")
processed_data['Month'] = processed_data['Date'].dt.to_period('M').astype(str)
orders_by_month = processed_data.groupby('Month')['Quantity_Ordered'].sum().reset_index()
fig3 = px.line(orders_by_month, x='Month', y='Quantity_Ordered', title='Total Quantity Ordered Over Time',
               markers=True, color_discrete_sequence=['#636EFA'])
st.plotly_chart(fig3)

st.subheader("Code for Chart 3")
st.code("""
processed_data['Month'] = processed_data['Date'].dt.to_period('M').astype(str)
orders_by_month = processed_data.groupby('Month')['Quantity_Ordered'].sum().reset_index()
fig3 = px.line(orders_by_month, x='Month', y='Quantity_Ordered', title='Total Quantity Ordered Over Time',
               markers=True, color_discrete_sequence=['#636EFA'])
st.plotly_chart(fig3)
""", language="python")

# Chart 4: Unit Price Distribution by Product
st.subheader("Chart 4: Unit Price Distribution by Product")
fig4 = px.box(processed_data, x='Product_Name', y='Unit_Price', title='Unit Price Distribution by Product',
              color='Product_Name', color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig4)

st.subheader("Code for Chart 4")
st.code("""
fig4 = px.box(processed_data, x='Product_Name', y='Unit_Price', title='Unit Price Distribution by Product',
              color='Product_Name', color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig4)
""", language="python")

# Chart 5: Stock Remaining vs. Quantity Ordered
st.subheader("Chart 5: Stock Remaining vs. Quantity Ordered")
fig5 = px.scatter(processed_data, x='Quantity_Ordered', y='Stock_Remaining', color='Product_Name',
                  title='Stock Remaining vs. Quantity Ordered', size='Total_Price',
                  color_discrete_sequence=px.colors.qualitative.Bold)
st.plotly_chart(fig5)

st.subheader("Code for Chart 5")
st.code("""
fig5 = px.scatter(processed_data, x='Quantity_Ordered', y='Stock_Remaining', color='Product_Name',
                  title='Stock Remaining vs. Quantity Ordered', size='Total_Price',
                  color_discrete_sequence=px.colors.qualitative.Bold)
st.plotly_chart(fig5)
""", language="python")

# --- Step 4: Model Selection and Training ---
st.header("Step 4: Model Selection and Training")
st.write("Given the dataset, a suitable task is to predict **Total_Price** based on features like Quantity_Ordered, Unit_Price, Stock_Remaining, and Product_Name. Since this is a regression task (predicting a continuous variable), a **Random Forest Regressor** is chosen for its robustness, ability to handle non-linear relationships, and feature importance insights.")

def train_model(data):
    # Prepare features and target
    X = data[['Quantity_Ordered', 'Unit_Price', 'Stock_Remaining']]
    X = pd.concat([X, pd.get_dummies(data['Product_Name'], prefix='Product')], axis=1)
    y = data['Total_Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

st.subheader("Code for Model Training")
st.code("""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model(data):
    X = data[['Quantity_Ordered', 'Unit_Price', 'Stock_Remaining']]
    X = pd.concat([X, pd.get_dummies(data['Product_Name'], prefix='Product')], axis=1)
    y = data['Total_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2
""", language="python")

# Train and display model results
model, mse, r2 = train_model(processed_data)
st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# --- Step 5: Save Processed Data ---
st.header("Step 5: Save Processed Data")
output = io.StringIO()
processed_data.to_csv(output, index=False)
st.download_button(
    label="Download Processed Data",
    data=output.getvalue(),
    file_name="processed_orders.csv",
    mime="text/csv"
)

st.subheader("Code for Saving Data")
st.code("""
output = io.StringIO()
processed_data.to_csv(output, index=False)
st.download_button(
    label="Download Processed Data",
    data=output.getvalue(),
    file_name="processed_orders.csv",
    mime="text/csv"
)
""", language="python")

if __name__ == "__main__":
    st.write("App executed successfully!")
