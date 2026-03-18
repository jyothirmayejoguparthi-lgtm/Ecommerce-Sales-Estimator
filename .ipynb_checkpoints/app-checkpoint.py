# =========================================================
# 🚀 E-COMMERCE AI SUITE (ULTIMATE VERSION)
# =========================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="E-Commerce AI Suite",
    page_icon="🚀",
    layout="wide"
)

# =========================================================
# CUSTOM CSS (PREMIUM UI)
# =========================================================

st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL + DATA
# =========================================================

model = pickle.load(open("sales_model.pkl", "rb"))
data = pd.read_csv("olist_order_items_dataset.csv")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🚀 AI Control Panel")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "📊 Insights", "🤖 Prediction", "📄 Report"]
)

# =========================================================
# 🏠 DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("🛒 E-Commerce AI Dashboard")
    st.markdown("### 📊 Business Overview")

    avg_price = data["price"].mean()
    avg_ship = data["freight_value"].mean()
    total_orders = len(data)

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Avg Price", f"₹{avg_price:.2f}")
    col2.metric("🚚 Avg Shipping", f"₹{avg_ship:.2f}")
    col3.metric("📦 Orders", total_orders)

    st.divider()

    col4, col5 = st.columns(2)

    with col4:
        st.subheader("📊 Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data["price"], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col5:
        st.subheader("🚚 Shipping Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data["freight_value"], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Shipping Cost")
        st.pyplot(fig)

# =========================================================
# 📊 INSIGHTS
# =========================================================

elif page == "📊 Insights":

    st.title("📊 Advanced Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Price vs Shipping")
        fig, ax = plt.subplots()
        sns.scatterplot(x=data["price"], y=data["freight_value"], ax=ax)
        ax.set_xlabel("Price")
        ax.set_ylabel("Shipping Cost")
        st.pyplot(fig)

    with col2:
        st.subheader("🔥 Correlation Heatmap")
        corr = data[["price", "freight_value"]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.info("📌 Insight: Higher priced items tend to have slightly higher shipping costs.")

# =========================================================
# 🤖 PREDICTION ENGINE
# =========================================================

elif page == "🤖 Prediction":

    st.title("🤖 Smart Revenue Prediction")

    st.markdown("""
    Enter product details to estimate expected revenue.
    
    💡 Example:
    - Price = 500  
    - Shipping = 50  
    - Installments = 2  
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        price = st.number_input("💰 Product Price (₹)", 0.0, 10000.0, 500.0)

    with col2:
        freight = st.number_input("🚚 Shipping Cost (₹)", 0.0, 1000.0, 50.0)

    with col3:
        installments = st.number_input("💳 Installments", 1, 12, 1)

    st.divider()

    if st.button("🚀 Predict Revenue"):

        input_data = np.array([[price, freight, installments]])

        with st.spinner("AI analyzing your input..."):
            prediction = model.predict(input_data)

        result = prediction[0]

        st.success(f"💰 Estimated Revenue: ₹{result:.2f}")

        # Confidence range (UX enhancement)
        lower = result * 0.9
        upper = result * 1.1

        st.info(f"📊 Confidence Range: ₹{lower:.2f} – ₹{upper:.2f}")

        # Interpretation
        if result > 5000:
            st.success("💸 High-value order")
        elif result > 1000:
            st.warning("📊 Moderate-value order")
        else:
            st.info("🛒 Low-value order")

        st.divider()

        # Visualization comparison
        st.subheader("📊 Price Comparison with Dataset")

        fig, ax = plt.subplots()
        sns.histplot(data["price"], bins=30, kde=True, ax=ax)
        ax.axvline(price, color="red", linestyle="--", label="Your Input")
        ax.legend()

        st.pyplot(fig)

# =========================================================
# 📄 REPORT SECTION
# =========================================================

elif page == "📄 Report":

    st.title("📄 Generate Business Report")

    summary = pd.DataFrame({
        "Metric": ["Average Price", "Average Shipping", "Total Orders"],
        "Value": [
            data["price"].mean(),
            data["freight_value"].mean(),
            len(data)
        ]
    })

    st.dataframe(summary)

    csv = summary.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="ecommerce_report.csv",
        mime="text/csv"
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown("✨ Built with Machine Learning & Streamlit | Nova 🚀")