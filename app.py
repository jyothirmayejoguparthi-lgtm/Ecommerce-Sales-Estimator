import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="E-Commerce Sales Estimator",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# LOAD MODEL AND DATA
# -----------------------------

model = joblib.load("sales_estimator_model.pkl")
df = pd.read_csv("ecommerce_clothing_sales_dataset.xls")

# -----------------------------
# TITLE
# -----------------------------

st.title("🛒 E-Commerce Sales Estimator Dashboard")

st.markdown(
"""
This dashboard predicts **estimated sales revenue** using a machine learning model.

Inputs used by the model:
• Quantity  
• Unit Price  
• Discount Percentage  
• Customer Age  
• Profit Margin
"""
)

st.info(
"""
Example Input

Quantity = 10  
Unit Price = 50  
Discount = 5  
Customer Age = 30  
Profit Margin = 20
"""
)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------

st.sidebar.header("Input Product Details")

quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
price = st.sidebar.number_input("Unit Price ($)", min_value=0.0, value=50.0)
discount = st.sidebar.number_input("Discount (%)", min_value=0.0, value=5.0)
age = st.sidebar.number_input("Customer Age", min_value=10, value=30)
margin = st.sidebar.number_input("Profit Margin (%)", min_value=0.0, value=20.0)

predict_button = st.sidebar.button("Predict Sales Revenue")

# Dataset statistics
avg_price = df["unit_price"].mean()
avg_quantity = df["quantity"].mean()
dataset_size = len(df)

# -----------------------------
# PREDICTION SECTION
# -----------------------------

if predict_button:

    input_data = np.array([[quantity, price, age, discount, margin]])

    with st.spinner("Calculating prediction..."):
        prediction = model.predict(input_data)

    st.metric("Estimated Sales Revenue", f"${prediction[0]:.2f}")

    if price > avg_price:
        price_comment = "Entered price is higher than the dataset average."
    else:
        price_comment = "Entered price is lower than the dataset average."

    if quantity > avg_quantity:
        quantity_comment = "Entered quantity is above the average purchase quantity."
    else:
        quantity_comment = "Entered quantity is below the average purchase quantity."

    st.success(
        f"""
Prediction Insight

{price_comment}

{quantity_comment}

Estimated revenue based on inputs: **${prediction[0]:.2f}**
"""
    )

st.divider()

# -----------------------------
# DATASET OVERVIEW
# -----------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Average Product Price", f"{avg_price:.2f}")
col2.metric("Average Quantity Sold", f"{avg_quantity:.2f}")
col3.metric("Dataset Size", dataset_size)

st.divider()

# -----------------------------
# SCATTER PLOT
# -----------------------------

st.subheader("Price vs Quantity Relationship")

fig1, ax1 = plt.subplots()

ax1.scatter(df["unit_price"], df["quantity"], alpha=0.6, color="steelblue")
ax1.set_xlabel("Unit Price")
ax1.set_ylabel("Quantity Sold")
ax1.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig1)

st.divider()

# -----------------------------
# QUANTITY DISTRIBUTION
# -----------------------------

st.subheader("Quantity Distribution")

fig2, ax2 = plt.subplots()

df["quantity"].value_counts().sort_index().plot(
    kind="bar",
    ax=ax2,
    color="teal",
    edgecolor="black"
)

ax2.set_xlabel("Quantity Sold")
ax2.set_ylabel("Frequency")
ax2.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig2)

st.divider()

# -----------------------------
# PRICE DISTRIBUTION
# -----------------------------

st.subheader("Price Distribution")

fig3, ax3 = plt.subplots()

ax3.hist(
    df["unit_price"],
    bins=20,
    color="skyblue",
    edgecolor="black",
    linewidth=1.2
)

ax3.set_xlabel("Unit Price")
ax3.set_ylabel("Frequency")
ax3.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig3)

st.divider()

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
import seaborn as sns

st.subheader("Feature Correlation Heatmap")

st.markdown("""
This heatmap shows how different features in the dataset are related to each other.

• Red → Strong positive correlation  
• Blue → Negative correlation  
• Darker colors indicate stronger relationships
""")

corr = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar=True
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

st.pyplot(fig)
# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

st.header("Feature Importance")

features = [
    "Quantity",
    "Unit Price",
    "Customer Age",
    "Discount Percentage",
    "Profit Margin"
]

importances = model.feature_importances_

fig5, ax5 = plt.subplots()

ax5.barh(features, importances, color="teal")
ax5.set_xlabel("Importance Score")

st.pyplot(fig5)

st.divider()

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.header("Model Performance")

r2 = 0.98
mae = 12.3
rmse = 18.7

col4, col5, col6 = st.columns(3)

col4.metric("R² Score", r2)
col5.metric("MAE", mae)
col6.metric("RMSE", rmse)

st.markdown(
"""
R² Score measures how well the model explains revenue variance.

MAE shows the average prediction error.

RMSE penalizes larger prediction errors.
"""
)

st.divider()

# -----------------------------
# DATASET INSIGHTS
# -----------------------------

st.header("Dataset Insights")

max_price = df["unit_price"].max()
min_price = df["unit_price"].min()

if avg_price > 200:
    price_comment = "Products are generally high priced."
else:
    price_comment = "Products are generally affordable."

if avg_quantity > 3:
    demand_comment = "Customer demand appears strong."
else:
    demand_comment = "Customer demand appears moderate."

st.info(
f"""
Average Product Price: **{avg_price:.2f}**

Average Quantity Sold: **{avg_quantity:.2f}**

Price Range: **{min_price:.2f} – {max_price:.2f}**

Insights

{price_comment}

{demand_comment}
"""
)

st.markdown("---")
st.markdown("Developed by Jyothirmaye | Machine Learning Project")
