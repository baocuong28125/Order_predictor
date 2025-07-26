import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Phân tích & Dự đoán Đơn hàng", layout="wide")

# Đường dẫn file
file_path = os.path.join(os.path.dirname(__file__), "orders_sample_with_stock.csv")

# Đọc dữ liệu
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Không tìm thấy file 'orders_sample_with_stock.csv'. Hãy chắc chắn rằng file này nằm cùng thư mục với app.py.")
    st.stop()

if df.empty:
    st.error("File dữ liệu rỗng hoặc không đọc được.")
    st.stop()

# Xử lý dữ liệu
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Order_Month'] = df['Date'].dt.month
le = LabelEncoder()
df['SKU_Code'] = le.fit_transform(df['SKU'])

# -----------------------
# 1. KHÁM PHÁ DỮ LIỆU
# -----------------------
st.title("📊 Phân tích dữ liệu đơn hàng")
st.subheader("1. Tổng quan dữ liệu")
st.write("Kích thước dữ liệu:", df.shape)
st.write("Các cột dữ liệu:", df.columns.tolist())
st.dataframe(df.head())

# -----------------------
# 2. TRỰC QUAN HÓA
# -----------------------
st.subheader("2. Trực quan hóa dữ liệu")

# Biểu đồ 1: Tổng Quantity theo SKU
st.markdown("### Biểu đồ 1: Tổng số lượng đặt hàng theo SKU")
fig1, ax1 = plt.subplots()
sku_quantity = df.groupby('SKU')['Quantity_Ordered'].sum().sort_values(ascending=False)
sns.barplot(x=sku_quantity.index, y=sku_quantity.values, ax=ax1)
ax1.set_title("Tổng Quantity theo SKU")
ax1.set_xlabel("SKU")
ax1.set_ylabel("Tổng Quantity")
st.pyplot(fig1)
st.code("""
fig1, ax1 = plt.subplots()
sku_quantity = df.groupby('SKU')['Quantity_Ordered'].sum().sort_values(ascending=False)
sns.barplot(x=sku_quantity.index, y=sku_quantity.values, ax=ax1)
ax1.set_title("Tổng Quantity theo SKU")
ax1.set_xlabel("SKU")
ax1.set_ylabel("Tổng Quantity")
st.pyplot(fig1)
""", language="python")

# Biểu đồ 2: Tồn kho còn lại theo SKU
st.markdown("### Biểu đồ 2: Tồn kho còn lại theo SKU")
fig2, ax2 = plt.subplots()
sku_stock = df.groupby('SKU')['Stock_Remaining'].mean()
sns.barplot(x=sku_stock.index, y=sku_stock.values, ax=ax2, palette="viridis")
ax2.set_title("Tồn kho trung bình theo SKU")
ax2.set_xlabel("SKU")
ax2.set_ylabel("Tồn kho trung bình")
st.pyplot(fig2)
st.code("""
fig2, ax2 = plt.subplots()
sku_stock = df.groupby('SKU')['Stock_Remaining'].mean()
sns.barplot(x=sku_stock.index, y=sku_stock.values, ax=ax2, palette="viridis")
ax2.set_title("Tồn kho trung bình theo SKU")
ax2.set_xlabel("SKU")
ax2.set_ylabel("Tồn kho trung bình")
st.pyplot(fig2)
""", language="python")

# Biểu đồ 3: Số lượng đơn hàng theo tháng
st.markdown("### Biểu đồ 3: Số lượng đơn hàng theo tháng")
fig3, ax3 = plt.subplots()
month_quantity = df.groupby('Order_Month')['Quantity_Ordered'].sum()
sns.lineplot(x=month_quantity.index, y=month_quantity.values, marker='o', ax=ax3)
ax3.set_title("Số lượng đặt hàng theo tháng")
ax3.set_xlabel("Tháng")
ax3.set_ylabel("Tổng Quantity")
st.pyplot(fig3)
st.code("""
fig3, ax3 = plt.subplots()
month_quantity = df.groupby('Order_Month')['Quantity_Ordered'].sum()
sns.lineplot(x=month_quantity.index, y=month_quantity.values, marker='o', ax=ax3)
ax3.set_title("Số lượng đặt hàng theo tháng")
ax3.set_xlabel("Tháng")
ax3.set_ylabel("Tổng Quantity")
st.pyplot(fig3)
""", language="python")

# Biểu đồ 4: Phân bố số lượng đặt hàng
st.markdown("### Biểu đồ 4: Phân bố số lượng đặt hàng (Histogram)")
fig4, ax4 = plt.subplots()
sns.histplot(df['Quantity_Ordered'], bins=20, kde=True, ax=ax4)
ax4.set_title("Phân bố số lượng đặt hàng")
ax4.set_xlabel("Quantity Ordered")
ax4.set_ylabel("Tần suất")
st.pyplot(fig4)
st.code("""
fig4, ax4 = plt.subplots()
sns.histplot(df['Quantity_Ordered'], bins=20, kde=True, ax=ax4)
ax4.set_title("Phân bố số lượng đặt hàng")
ax4.set_xlabel("Quantity Ordered")
ax4.set_ylabel("Tần suất")
st.pyplot(fig4)
""", language="python")

# Biểu đồ 5: Heatmap tương quan
st.markdown("### Biểu đồ 5: Ma trận tương quan giữa các biến")
fig5, ax5 = plt.subplots(figsize=(6,4))
corr = df[['Quantity_Ordered', 'Stock_Remaining', 'Order_Month', 'SKU_Code']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
ax5.set_title("Heatmap tương quan giữa các biến")
st.pyplot(fig5)
st.code("""
fig5, ax5 = plt.subplots(figsize=(6,4))
corr = df[['Quantity_Ordered', 'Stock_Remaining', 'Order_Month', 'SKU_Code']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
ax5.set_title("Heatmap tương quan giữa các biến")
st.pyplot(fig5)
""", language="python")

# -----------------------
# 3. MÔ HÌNH DỰ ĐOÁN
# -----------------------
st.title("🤖 Dự đoán số lượng đơn hàng")

# Tạo X, y
X = df[['SKU_Code', 'Stock_Remaining', 'Order_Month']]
y = df['Quantity_Ordered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Giao diện dự đoán
sku_input = st.selectbox("Chọn sản phẩm", df['SKU'].unique())
stock_input = st.slider("Tồn kho hiện tại", 0, 100, 10)
month_input = st.slider("Tháng đặt hàng", 1, 12, 6)

# Dự đoán
sku_code = le.transform([sku_input])[0]
input_data = [[sku_code, stock_input, month_input]]
predicted_qty = model.predict(input_data)[0]

st.subheader("Kết quả dự đoán:")
st.write(f"Số lượng dự đoán: **{int(predicted_qty)}** đơn vị")
