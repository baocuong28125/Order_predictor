# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Order Quantity Predictor", layout="centered")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/baocuong28125/Order_predictor/refs/heads/main/orders_sample_with_stock.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['Order_Month'] = df['Date'].dt.month
    return df

df = load_data()

# Encode SKU
le = LabelEncoder()
df['SKU_Code'] = le.fit_transform(df['SKU'])

# Train model
X = df[['SKU_Code', 'Stock_Remaining', 'Order_Month']]
y = df['Quantity_Ordered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("📦 Dự đoán số lượng đặt hàng")

sku_input = st.selectbox("Chọn sản phẩm (SKU)", df['SKU'].unique())
stock_input = st.slider("Tồn kho còn lại", min_value=0, max_value=100, value=20)
month_input = st.slider("Tháng đặt hàng", min_value=1, max_value=12, value=6)

sku_code = le.transform([sku_input])[0]
input_df = pd.DataFrame([[sku_code, stock_input, month_input]], columns=X.columns)

prediction = model.predict(input_df)[0]

st.markdown(f"### 👉 Dự đoán số lượng đặt hàng: **{round(prediction)} sản phẩm**")
