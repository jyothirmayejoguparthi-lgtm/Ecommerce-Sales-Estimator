st.write("NEW VERSION RUNNING")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Ecommerce Sales Estimator", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("🛒 Ecommerce Sales Estimator")
st.markdown("Analyze sales data, understand trends, and predict revenue using Machine Learning.")

st.markdown("---")

# -----------------------------
# LOAD DATA
# -----------------------------
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
st.markdown("## 📊 Data Insights Dashboard")
st.markdown("Understand pricing patterns, shipping trends, and relationships between variables.")

st.markdown("---")

# ROW 1
col1, col2 = st.columns(2)

# PRICE DISTRIBUTION
with col1:
    st.markdown("### 🛒 Price Distribution Analysis")
    st.write("Shows how product prices are spread across all orders.")

    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.histplot(df['price'], bins=50, ax=ax1)
    ax1.set_title("Product Price Distribution")
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Frequency")

    st.pyplot(fig1)

    st.success("Insight: Most products fall in the low price range, with few expensive outliers.")

# SHIPPING DISTRIBUTION
with col2:
    st.markdown("### 🚚 Shipping Cost Analysis")
    st.write("Displays variation in shipping costs across orders.")

    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.histplot(df['freight_value'], bins=50, ax=ax2)
    ax2.set_title("Shipping Cost Distribution")
    ax2.set_xlabel("Shipping Cost")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2)

    st.success("Insight: Shipping costs are generally low but have some high-cost exceptions.")

st.markdown("---")

# SCATTER PLOT
st.markdown("### 📈 Price vs Shipping Relationship")
st.write("Analyzes how product price affects shipping cost.")

fig3, ax3 = plt.subplots(figsize=(8,4))
sns.scatterplot(x=df['price'], y=df['freight_value'], ax=ax3)
ax3.set_xlabel("Price")
ax3.set_ylabel("Shipping Cost")

st.pyplot(fig3)

st.success("Insight: Higher priced items tend to have slightly higher shipping costs.")

st.markdown("---")

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.markdown("## 🤖 Model Performance & Reliability")

st.write("These metrics help evaluate how accurate and reliable the prediction models are.")

perf_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "R² Score": [0.56, 0.79, 0.70],
    "MAE": ["-", "54.15", "-"],
    "RMSE": ["-", "139.84", "-"]
})

st.dataframe(perf_df, use_container_width=True)

st.success("Conclusion: Random Forest performs best with highest accuracy (R² ≈ 0.79).")

st.markdown("---")

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.markdown("## 🔮 Revenue Prediction")

st.write("Enter product details to estimate expected revenue.")

col1, col2, col3 = st.columns(3)

price = col1.number_input("Product Price (₹)", value=500.0)
shipping = col2.number_input("Shipping Cost (₹)", value=50.0)
installments = col3.number_input("Installments", value=1)

if st.button("Predict Revenue"):
    model = pickle.load(open("sales_model.pkl", "rb"))
    prediction = model.predict([[price, shipping, installments]])

    st.success(f"💰 Estimated Revenue: ₹{prediction[0]:.2f}")

st.markdown("---")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("Built with Machine Learning & Streamlit")
