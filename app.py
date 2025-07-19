import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set layout
st.set_page_config(page_title="Phân tích & Dự đoán Đơn hàng", layout="wide")
st.title("📦 Ứng dụng Phân tích & Dự đoán Đơn hàng")

# ✅ Đọc dữ liệu từ GitHub
url = "https://raw.githubusercontent.com/baocuong28125/Order_predictor/main/orders_sample_with_stock.csv"
df = pd.read_csv(url)
df.columns = df.columns.str.strip().str.lower()

# ✅ Tiền xử lý
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['Order_Month'] = df['Date'].dt.month
df['Total_Revenue'] = df['Quantity_Ordered'] * df['Unit_Price']
le = LabelEncoder()
df['SKU_Code'] = le.fit_transform(df['SKU'])

# Menu chức năng
menu = st.sidebar.radio("Chọn chức năng", ["📊 Trực quan hóa dữ liệu", "🧹 Tiền xử lý", "🤖 Mô hình dự đoán"])

if menu == "📊 Trực quan hóa dữ liệu":
    st.subheader("1. Tổng số lượng đặt hàng theo tháng")
    st.code(
        """
        fig1, ax1 = plt.subplots()
        sns.barplot(x='Order_Month', y='Quantity_Ordered', data=df, ax=ax1)
        st.pyplot(fig1)
        """,
        language='python')
    fig1, ax1 = plt.subplots()
    sns.barplot(x='Order_Month', y='Quantity_Ordered', data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("2. Tổng đặt hàng theo SKU")
    st.code(
        """
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            df.groupby('SKU')['Quantity_Ordered'].sum().sort_values(ascending=False).plot(kind='bar', ax=ax2)
            st.pyplot(fig2)
            """,
            language='python')
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df.groupby('SKU')['Quantity_Ordered'].sum().sort_values(ascending=False).plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Trung bình tồn kho theo SKU")
    st.code(
        """
    fig3, ax3 = plt.subplots()
    df.groupby('SKU')['Stock_Remaining'].mean().plot(kind='barh', ax=ax3)
    st.pyplot(fig3)
    """,
            language='python')
    fig3, ax3 = plt.subplots()
    df.groupby('SKU')['Stock_Remaining'].mean().plot(kind='barh', ax=ax3)
    st.pyplot(fig3)

    st.subheader("4. Phân phối số lượng đặt hàng")
    st.code(
        """
    fig4, ax4 = plt.subplots() sns.histplot(
    df['Quantity_Ordered'], kde=True, bins=20, ax=ax4) 
    st.pyplot(fig4)
    """,
            language='python')
    fig4, ax4 = plt.subplots()
    sns.histplot(df['Quantity_Ordered'], kde=True, bins=20, ax=ax4)
    st.pyplot(fig4)

    st.subheader("5. Doanh thu theo sản phẩm")
    st.code(
        """
    fig5, ax5 = plt.subplots() 
    df.groupby('Product_Name')['Total_Revenue'].sum().sort_values().plot(kind='barh', ax=ax5) 
    st.pyplot(fig5)
    """,
            language='python')
    fig5, ax5 = plt.subplots()
    df.groupby('Product_Name')['Total_Revenue'].sum().sort_values().plot(kind='barh', ax=ax5)
    st.pyplot(fig5)

elif menu == "🧹 Tiền xử lý":
    st.subheader("Thông tin dữ liệu đầu vào")
    st.code("df.head()", language='python')
    st.write(df.head())

    st.subheader("Giá trị thiếu")
    st.code("df.isnull().sum()",language="python")
    st.write(df.isnull().sum())

    st.subheader("Mã hóa SKU → SKU_Code")
    st.code("df[['SKU', 'SKU_Code']].drop_duplicates()", language='python')
    st.write(df[['SKU', 'SKU_Code']].drop_duplicates())

    st.subheader("Các biến tạo mới")
    st.markdown("- Order_Month")
    st.markdown("- Total_Revenue")

    st.subheader("Thống kê mô tả")
    st.code("df.describe()",language='python')
    st.write(df.describe())

elif menu == "🤖 Mô hình dự đoán":
st.title("Ứng dụng Dự đoán Đơn hàng với Random Forest")

# Hiển thị mã huấn luyện bằng st.code
st.subheader("Mã Python: Huấn luyện mô hình Random Forest")
rf_code = '''
X = df[['sku_code', 'stock_remaining', 'order_month']]
y = df['quantity_ordered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
'''
st.code(rf_code, language='python')

# Huấn luyện mô hình
X = df[['sku_code', 'stock_remaining', 'order_month']]
y = df['quantity_ordered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# Hiển thị kết quả
st.subheader("Kết quả đánh giá mô hình")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.3f}")

# Trực quan hóa kết quả dự đoán
st.subheader("Biểu đồ: Thực tế vs Dự đoán")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Giá trị Thực tế")
ax.set_ylabel("Giá trị Dự đoán")
ax.set_title("So sánh Giá trị Thực tế và Dự đoán")
st.pyplot(fig)
