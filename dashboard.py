import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from churn_model import train_churn_model, load_or_train_model, plot_churn_distribution
import joblib
from churn_model import predict_churn
from clv_model import prepare_rfm_data, fit_and_predict_clv

# Set Streamlit page configuration
st.set_page_config(page_title="Customer Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1MaKNYEzplmTvSjJ2BPZqbbDn72LswnNY"
    retail_df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
    rfm_df = pd.read_csv("RFM1.csv")
    return retail_df, rfm_df

retail_df, rfm_df = load_data()

# Sidebar Navigation with selectbox
st.sidebar.title("📊 Customer Analytics")
selected_module = st.sidebar.selectbox("Select Module", ["Overview", "Churn Analysis", "RFM Analysis", "Churn Prediction"])

# ================= OVERVIEW =================
if selected_module == "Overview":
    st.title("🛍️ Online Retail Dashboard")
    st.markdown("<h2 style='color: #0072B5; font-size: 40px;'>Customer Analytics Dashboard</h2>", unsafe_allow_html=True)

    # KPIs
    st.subheader("Key Performance Indicators")
    total_revenue = (retail_df['Quantity'] * retail_df['UnitPrice']).sum()
    unique_customers = retail_df['CustomerID'].nunique()
    top_country = retail_df['Country'].value_counts().idxmax()

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("👥 Unique Customers", unique_customers)
    col3.metric("🌍 Top Country", top_country)

    # Show sample data
    st.subheader("Sample Retail Data")
    st.dataframe(retail_df.head())
    
    st.title("Power BI Dashboard Preview")

    # Show dashboard image
    st.image("EDA.png", caption="Dashboard Preview", use_column_width=True)

# ================= CHURN ANALYSIS =================
elif selected_module == "Churn Analysis":
    st.title("Churn Analysis")
    st.markdown("""
    This section will provide insights on customer churn trends.  
    """)

    # Example: Number of transactions per customer
    st.subheader("Transactions per Customer")
    cust_orders = retail_df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    cust_orders.columns = ['CustomerID', 'NumTransactions']

    fig, ax = plt.subplots()
    sns.histplot(cust_orders['NumTransactions'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Customer Transactions")
    ax.set_xlabel("Number of Transactions")
    st.pyplot(fig)

    uploaded_file = st.file_uploader("Upload Data with 'Churn' column", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Churn' in df.columns:
            fig = plot_churn_distribution(df)
            st.pyplot(fig)
        else:
            st.error("Column 'Churn' not found in uploaded file")
# ================= RFM ANALYSIS =================
elif selected_module == "RFM Analysis":
    st.title("RFM Segmentation & Scores")

    st.subheader("RFM Data")
    
    # Clean the data before displaying or processing
    rfm_df = rfm_df.dropna(subset=['CustomerID'])

    # Optional: Remove zero or negative values
    rfm_df = rfm_df[
        (rfm_df['Recency'] > 0) &
        (rfm_df['Frequency'] > 0) &
        (rfm_df['Monetary'] > 0)
    ]
    st.dataframe(rfm_df.head())
        
    # Segment distribution
    st.subheader("Customer Segment Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=rfm_df, x='CustomerSegment', order=rfm_df['CustomerSegment'].value_counts().index, palette='viridis', ax=ax1)
    ax1.set_title("Customer Segments")
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    
    
    # RFM Histograms
    col4, col5, col6 = st.columns(3)
    with col4:
        fig2, ax2 = plt.subplots()
        sns.histplot(rfm_df['Recency'], bins=20, kde=True, color='skyblue', ax=ax2)
        ax2.set_title("Recency")
        st.pyplot(fig2)

    with col5:
        fig3, ax3 = plt.subplots()
        sns.histplot(rfm_df['Frequency'], bins=20, kde=True, color='lightgreen', ax=ax3)
        ax3.set_title("Frequency")
        st.pyplot(fig3)

    with col6:
        fig4, ax4 = plt.subplots()
        sns.histplot(rfm_df['Monetary'], bins=20, kde=True, color='salmon', ax=ax4)
        ax4.set_title("Monetary")
        st.pyplot(fig4)

    # RFM Heatmap
    st.subheader("Heatmap: MonetaryScore by R & F")
    rfm_pivot = rfm_df.pivot_table(index='RecencyScore', columns='FrequencyScore', values='MonetaryScore', aggfunc='mean')
    fig5, ax5 = plt.subplots()
    sns.heatmap(rfm_pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax5)
    ax5.set_title("Avg Monetary Score")
    st.pyplot(fig5)

# ================= CHURN PREDICTION =================
elif selected_module == "Churn Prediction":
    st.title("Churn Prediction")
    st.markdown("""
    This section is for **predicting churn using machine learning models**.
    """)
    st.subheader("Enter Customer Details")
    recency = st.number_input("Recency (days)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0)
    input_data = [recency, frequency, monetary]
    if st.button("Predict Churn"):
        input_data = [recency, frequency, monetary]
        result = predict_churn(input_data)
        st.write("Prediction result:", result)  # Add this line for debugging
        if result[0] == 1:
            st.error("⚠️ This customer is likely to **churn**.")
        else:
            st.success("✅ This customer is likely to stay active.")

retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'], errors='coerce')
retail_df = retail_df[
    (retail_df['CustomerID'].notnull()) &
    (retail_df['Quantity'] > 0) &
    (retail_df['UnitPrice'] > 0)
]

rfm = prepare_rfm_data(retail_df)
clv_data = fit_and_predict_clv(rfm)
