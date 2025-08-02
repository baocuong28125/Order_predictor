import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân tích & Dự đoán Đơn hàng", layout="wide")

# -----------------------
# 1. XỬ LÝ DỮ LIỆU
# -----------------------
# Đường dẫn file
file_path = os.path.join(os.path.dirname(__file__), "orders_sample_with_stock.csv")

# Đọc dữ liệu
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Không tìm thấy file 'orders_sample_with_stock.csv'. Hãy chắc chắn rằng file này nằm cùng thư mục với app.py.")
    st.stop()

# Kiểm tra dữ liệu rỗng
if df.empty:
    st.error("File dữ liệu rỗng hoặc không đọc được.")
    st.stop()

# Xử lý dữ liệu
# Chuyển đổi cột Date sang định dạng datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

# Tạo cột Order_Month từ cột Date
df['Order_Month'] = df['Date'].dt.month

# Xử lý giá trị thiếu
df = df.dropna(subset=['Date', 'Order_Month', 'SKU', 'Quantity_Ordered', 'Stock_Remaining', 'Unit_Price'])

# Xóa các bản ghi trùng lặp
df = df.drop_duplicates()

# Kiểm tra và chuẩn hóa dữ liệu số
df['Quantity_Ordered'] = pd.to_numeric(df['Quantity_Ordered'], errors='coerce').fillna(0).astype(int)
df['Stock_Remaining'] = pd.to_numeric(df['Stock_Remaining'], errors='coerce').fillna(0).astype(int)
df['Unit_Price'] = pd.to_numeric(df['Unit_Price'], errors='coerce').fillna(0).astype(float)

# Mã hóa cột SKU thành giá trị số
le = LabelEncoder()
df['SKU_Code'] = le.fit_transform(df['SKU'])

# -----------------------
# 2. KHÁM PHÁ DỮ LIỆU
# -----------------------
st.title("📊 Phân tích dữ liệu đơn hàng")
st.subheader("1. Tổng quan dữ liệu")
st.write("Kích thước dữ liệu:", df.shape)
st.write("Các cột dữ liệu:", df.columns.tolist())
st.dataframe(df.head())

# -----------------------
# 3. TRỰC QUAN HÓA
# -----------------------
st.subheader("2. Trực quan hóa dữ liệu")

# Biểu đồ 1: Tổng Quantity theo SKU
st.markdown("### Biểu đồ 1: Tổng số lượng đặt hàng theo SKU")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sku_quantity = df.groupby('SKU')['Quantity_Ordered'].sum().sort_values(ascending=False)
sns.barplot(x=sku_quantity.index, y=sku_quantity.values, ax=ax1, palette="muted")
ax1.set_title("Tổng Quantity theo SKU", fontsize=14)
ax1.set_xlabel("SKU", fontsize=12)
ax1.set_ylabel("Tổng Quantity", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig1)
st.code("""
fig1, ax1 = plt.subplots(figsize=(10, 6))
sku_quantity = df.groupby('SKU')['Quantity_Ordered'].sum().sort_values(ascending=False)
sns.barplot(x=sku_quantity.index, y=sku_quantity.values, ax=ax1, palette="muted")
ax1.set_title("Tổng Quantity theo SKU", fontsize=14)
ax1.set_xlabel("SKU", fontsize=12)
ax1.set_ylabel("Tổng Quantity", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig1)
""", language="python")

# Biểu đồ 2: Tồn kho còn lại theo SKU
st.markdown("### Biểu đồ 2: Tồn kho còn lại theo SKU")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sku_stock = df.groupby('SKU')['Stock_Remaining'].mean()
sns.barplot(x=sku_stock.index, y=sku_stock.values, ax=ax2, palette="viridis")
ax2.set_title("Tồn kho trung bình theo SKU", fontsize=14)
ax2.set_xlabel("SKU", fontsize=12)
ax2.set_ylabel("Tồn kho trung bình", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig2)
st.code("""
fig2, ax2 = plt.subplots(figsize=(10, 6))
sku_stock = df.groupby('SKU')['Stock_Remaining'].mean()
sns.barplot(x=sku_stock.index, y=sku_stock.values, ax=ax2, palette="viridis")
ax2.set_title("Tồn kho trung bình theo SKU", fontsize=14)
ax2.set_xlabel("SKU", fontsize=12)
ax2.set_ylabel("Tồn kho trung bình", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig2)
""", language="python")

# Biểu đồ 3: Số lượng đơn hàng theo tháng
st.markdown("### Biểu đồ 3: Số lượng đơn hàng theo tháng")
fig3, ax3 = plt.subplots(figsize=(10, 6))
month_quantity = df.groupby('Order_Month')['Quantity_Ordered'].sum()
sns.lineplot(x=month_quantity.index, y=month_quantity.values, marker='o', ax=ax3, color='b')
ax3.set_title("Số lượng đặt hàng theo tháng", fontsize=14)
ax3.set_xlabel("Tháng", fontsize=12)
ax3.set_ylabel("Tổng Quantity", fontsize=12)
st.pyplot(fig3)
st.code("""
fig3, ax3 = plt.subplots(figsize=(10, 6))
month_quantity = df.groupby('Order_Month')['Quantity_Ordered'].sum()
sns.lineplot(x=month_quantity.index, y=month_quantity.values, marker='o', ax=ax3, color='b')
ax3.set_title("Số lượng đặt hàng theo tháng", fontsize=14)
ax3.set_xlabel("Tháng", fontsize=12)
ax3.set_ylabel("Tổng Quantity", fontsize=12)
st.pyplot(fig3)
""", language="python")

# Biểu đồ 4: Phân bố số lượng đặt hàng
st.markdown("### Biểu đồ 4: Phân bố số lượng đặt hàng (Histogram)")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.histplot(df['Quantity_Ordered'], bins=20, kde=True, ax=ax4, color='g')
ax4.set_title("Phân bố số lượng đặt hàng", fontsize=14)
ax4.set_xlabel("Quantity Ordered", fontsize=12)
ax4.set_ylabel("Tần suất", fontsize=12)
st.pyplot(fig4)
st.code("""
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.histplot(df['Quantity_Ordered'], bins=20, kde=True, ax=ax4, color='g')
ax4.set_title("Phân bố số lượng đặt hàng", fontsize=14)
ax4.set_xlabel("Quantity Ordered", fontsize=12)
ax4.set_ylabel("Tần suất", fontsize=12)
st.pyplot(fig4)
""", language="python")

# Biểu đồ 5: Heatmap tương quan
st.markdown("### Biểu đồ 5: Ma trận tương quan giữa các biến")
fig5, ax5 = plt.subplots(figsize=(8, 6))
corr = df[['Quantity_Ordered', 'Stock_Remaining', 'Order_Month', 'SKU_Code', 'Unit_Price']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5, fmt=".2f")
ax5.set_title("Heatmap tương quan giữa các biến", fontsize=14)
st.pyplot(fig5)
st.code("""
fig5, ax5 = plt.subplots(figsize=(8, 6))
corr = df[['Quantity_Ordered', 'Stock_Remaining', 'Order_Month', 'SKU_Code', 'Unit_Price']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5, fmt=".2f")
ax5.set_title("Heatmap tương quan giữa các biến", fontsize=14)
st.pyplot(fig5)
""", language="python")

# -----------------------
# 4. MÔ HÌNH DỰ ĐOÁN
# -----------------------
st.title("🤖 Dự đoán số lượng đơn hàng")

# Tạo X, y cho mô hình
X = df[['SKU_Code', 'Stock_Remaining', 'Order_Month', 'Unit_Price']]
y = df['Quantity_Ordered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Giao diện nhập liệu cho dự đoán
st.subheader("Nhập thông tin để dự đoán")
sku_input = st.selectbox("Chọn sản phẩm (SKU)", df['SKU'].unique())
stock_input = st.slider("Tồn kho hiện tại", min_value=0, max_value=100, value=10)
month_input = st.slider("Tháng đặt hàng", min_value=1, max_value=12, value=6)
unit_price_input = st.slider("Đơn giá", min_value=0.0, max_value=2000.0, value=500.0)

# Dự đoán
sku_code = le.transform([sku_input])[0]
input_data = [[sku_code, stock_input, month_input, unit_price_input]]
predicted_qty = model.predict(input_data)[0]

# Hiển thị kết quả
st.subheader("Kết quả dự đoán:")
st.write(f"Số lượng dự đoán: **{int(predicted_qty)}** đơn vị")
st.code("""
sku_code = le.transform([sku_input])[0]
input_data = [[sku_code, stock_input, month_input, unit_price_input]]
predicted_qty = model.predict(input_data)[0]
st.write(f"Số lượng dự đoán: **{int(predicted_qty)}** đơn vị")
""", language="python")
