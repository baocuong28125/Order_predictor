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
    code_visualization_2 = '''
st.subheader("2. Tổng đặt hàng theo SKU")
fig2, ax2 = plt.subplots(figsize=(10, 4))
df.groupby('SKU')['Quantity_Ordered'].sum().sort_values(ascending=False).plot(kind='bar', ax=ax2)
st.pyplot(fig2)
'''

st.code(code_visualization_2, language='python')
    fig1, ax1 = plt.subplots()
    sns.barplot(x='Order_Month', y='Quantity_Ordered', data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("2. Tổng đặt hàng theo SKU")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df.groupby('SKU')['Quantity_Ordered'].sum().sort_values(ascending=False).plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Trung bình tồn kho theo SKU")
    fig3, ax3 = plt.subplots()
    df.groupby('SKU')['Stock_Remaining'].mean().plot(kind='barh', ax=ax3)
    st.pyplot(fig3)

    st.subheader("4. Phân phối số lượng đặt hàng")
    fig4, ax4 = plt.subplots()
    sns.histplot(df['Quantity_Ordered'], kde=True, bins=20, ax=ax4)
    st.pyplot(fig4)

    st.subheader("5. Doanh thu theo sản phẩm")
    fig5, ax5 = plt.subplots()
    df.groupby('Product_Name')['Total_Revenue'].sum().sort_values().plot(kind='barh', ax=ax5)
    st.pyplot(fig5)

elif menu == "🧹 Tiền xử lý":
    st.subheader("Thông tin dữ liệu đầu vào")
    st.code('''df.head()''', language='python')
    st.write(df.head())

    st.subheader("Giá trị thiếu")
    st.code("""df.isnull().sum()""",language="python")
    st.write(df.isnull().sum())

    st.subheader("Mã hóa SKU → SKU_Code")
    st.code('''df[['SKU', 'SKU_Code']].drop_duplicates()''', language='python')
    st.write(df[['SKU', 'SKU_Code']].drop_duplicates())

    st.subheader("Các biến tạo mới")
    st.markdown("- Order_Month")
    st.markdown("- Total_Revenue")

    st.subheader("Thống kê mô tả")
    st.code('''df.describe()''', language='python')
    st.write(df.describe())

elif menu == "🤖 Mô hình dự đoán":
    st.subheader("Huấn luyện & đánh giá mô hình")

    # Dữ liệu đầu vào
    X = df[['SKU_Code', 'Stock_Remaining', 'Order_Month']]
    y = df['Quantity_Ordered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5  # ✅ sửa lỗi ở đây
        r2 = r2_score(y_test, y_pred)
        results.append((name, round(mae, 2), round(rmse, 2), round(r2, 3)))

    results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R²"])
    st.dataframe(results_df.sort_values("R²", ascending=False))
