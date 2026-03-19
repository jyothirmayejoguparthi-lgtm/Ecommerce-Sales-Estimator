import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Ecommerce Sales Estimator", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("🛒 Ecommerce Sales Estimator")
st.markdown("Analyze sales data, visualize trends, and predict revenue using Machine Learning.")

st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🔮 Prediction Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "Linear Regression"]
)

user_price = st.sidebar.number_input("Product Price (₹)", value=500.0)
user_shipping = st.sidebar.number_input("Shipping Cost (₹)", value=50.0)
user_installments = st.sidebar.number_input("Installments", value=1)

predict_btn = st.sidebar.button("Predict Revenue")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV (Optional)", type=["csv"])

# -----------------------------
# LOAD DATA
# -----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset loaded!")
else:
    orders = pd.read_csv("olist_orders_dataset.csv")
    items = pd.read_csv("olist_order_items_dataset.csv")
    payments = pd.read_csv("olist_order_payments_dataset.csv")

    df = items.merge(orders, on="order_id")
    df = df.merge(payments, on="order_id")

# -----------------------------
# METRICS DASHBOARD
# -----------------------------
avg_price = df['price'].mean()
avg_shipping = df['freight_value'].mean()
total_orders = df['order_id'].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("🛒 Avg Price", f"₹{avg_price:.2f}")
col2.metric("🚚 Avg Shipping", f"₹{avg_shipping:.2f}")
col3.metric("📦 Total Orders", total_orders)

st.markdown("---")

# -----------------------------
# DATA INSIGHTS
# -----------------------------
st.header("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🛒 Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['price'], bins=50, ax=ax1)
    st.pyplot(fig1)
    st.info("👉 Most products are low priced.")

with col2:
    st.subheader("🚚 Shipping Cost Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['freight_value'], bins=50, ax=ax2)
    st.pyplot(fig2)
    st.info("👉 Most shipping costs are low.")

# Scatter
st.subheader("📈 Price vs Shipping Cost")
fig3, ax3 = plt.subplots()
sns.scatterplot(x=df['price'], y=df['freight_value'], ax=ax3)
st.pyplot(fig3)

# Heatmap
st.subheader("🔥 Correlation Heatmap")
numeric_df = df[['price', 'freight_value', 'payment_value', 'payment_installments']]
corr = numeric_df.corr()

fig4, ax4 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

st.markdown("---")

# -----------------------------
# MACHINE LEARNING
# -----------------------------
X = df[['price', 'freight_value', 'payment_installments']]
y = df['payment_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Metrics
lr_r2 = r2_score(y_test, lr_pred)
rf_r2 = r2_score(y_test, rf_pred)

lr_mae = mean_absolute_error(y_test, lr_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

lr_rmse = root_mean_squared_error(y_test, lr_pred)
rf_rmse = root_mean_squared_error(y_test, rf_pred)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.header("🤖 Model Performance")

perf_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [round(lr_r2,2), round(rf_r2,2)],
    "MAE": [round(lr_mae,2), round(rf_mae,2)],
    "RMSE": [round(lr_rmse,2), round(rf_rmse,2)]
})

st.table(perf_df)
st.info("👉 Random Forest performs better for this dataset.")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📊 Feature Importance (Random Forest)")

importance = rf.feature_importances_

fig5, ax5 = plt.subplots()
sns.barplot(x=importance, y=X.columns, ax=ax5)
ax5.set_title("Feature Importance")
st.pyplot(fig5)

st.markdown("---")

# -----------------------------
# PREDICTION OUTPUT
# -----------------------------
st.header("💰 Predicted Revenue")

if predict_btn:
    if model_choice == "Random Forest":
        prediction = rf.predict([[user_price, user_shipping, user_installments]])
    else:
        prediction = lr.predict([[user_price, user_shipping, user_installments]])

    value = prediction[0]

    st.markdown(f"""
    <div style="
        background-color:#1f77b4;
        padding:20px;
        border-radius:10px;
        text-align:center;
        color:white;
        font-size:30px;
        font-weight:bold;">
        ₹ {value:,.2f}
    </div>
    """, unsafe_allow_html=True)

    lower = value * 0.9
    upper = value * 1.1

    st.info(f"📊 Expected Range: ₹{lower:,.2f} – ₹{upper:,.2f}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built with Machine Learning & Streamlit")