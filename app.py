import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Dự đoán Đơn hàng", layout="centered")

# Xác định đường dẫn file CSV (cùng thư mục với app.py)
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

# Kiểm tra cột Date
if 'Date' not in df.columns:
    st.error("Không tìm thấy cột 'Date' trong dữ liệu. Vui lòng kiểm tra lại file.")
    st.write("Các cột hiện có:", df.columns.tolist())
    st.stop()

# Chuyển đổi kiểu dữ liệu ngày
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
if df['Date'].isna().all():
    st.error("Không thể chuyển đổi giá trị trong cột 'Date' sang định dạng ngày tháng.")
    st.stop()

# Thêm cột tháng
df['Order_Month'] = df['Date'].dt.month

# Mã hóa SKU
if 'SKU' not in df.columns:
    st.error("Không tìm thấy cột 'SKU' trong dữ liệu.")
    st.stop()

le = LabelEncoder()
df['SKU_Code'] = le.fit_transform(df['SKU'])

# Kiểm tra cột Quantity_Ordered & Stock_Remaining
required_cols = ['Quantity_Ordered', 'Stock_Remaining']
for col in required_cols:
    if col not in df.columns:
        st.error(f"Không tìm thấy cột '{col}' trong dữ liệu.")
        st.stop()

# Tạo X, y
X = df[['SKU_Code', 'Stock_Remaining', 'Order_Month']]
y = df['Quantity_Ordered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Giao diện
st.title("Ứng dụng Phân tích & Dự đoán Đơn hàng")

# Hiển thị preview dữ liệu
with st.expander("Xem dữ liệu đầu vào"):
    st.dataframe(df.head())

sku_input = st.selectbox("Chọn sản phẩm", df['SKU'].unique())
stock_input = st.slider("Tồn kho hiện tại", 0, 100, 10)
month_input = st.slider("Tháng đặt hàng", 1, 12, 6)

# Dự đoán
sku_code = le.transform([sku_input])[0]
input_data = [[sku_code, stock_input, month_input]]
predicted_qty = model.predict(input_data)[0]

st.subheader("Kết quả dự đoán:")
st.write(f"Số lượng dự đoán: **{int(predicted_qty)}** đơn vị")
