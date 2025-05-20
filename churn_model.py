import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import shap
import joblib
import os

# =========================
# Step 1: Train and Save Model
# =========================

def train_churn_model():
    print("Training churn model...")

    # Load data
    rfm_df = pd.read_csv('Online_retail.csv', encoding='ISO-8859-1')
    rfm_df['InvoiceDate'] = pd.to_datetime(rfm_df['InvoiceDate'], errors='coerce')

    reference_date = rfm_df['InvoiceDate'].max()
    recency_info = rfm_df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency_info['DaysSinceLastPurchase'] = (reference_date - recency_info['InvoiceDate']).dt.days
    recency_info['Churn'] = recency_info['DaysSinceLastPurchase'].apply(lambda x: 1 if x > 180 else 0)

    rfm1_df = pd.read_csv('RFM1.csv')
    df = pd.merge(rfm1_df, recency_info[['CustomerID', 'Churn']], on='CustomerID', how='inner')

    X = df[['Recency', 'Frequency', 'Monetary']]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'churn_model.pkl')
    print("Model trained and saved as churn_model.pkl")  # ✅ removed
    # or, use ASCII:
    print("[OK] Model trained and saved as churn_model.pkl")


    return model, df, X_test


# =========================
# Step 2: Load or Train Model
# =========================

def load_or_train_model():
    if os.path.exists('churn_model.pkl'):
        print("📦 Loading existing model...")
        model = joblib.load('churn_model.pkl')
    else:
        model, _, _ = train_churn_model()
    return model


# =========================
# Step 3: Plotting Function
# =========================

def plot_churn_distribution(df):
    churn_counts = df['Churn'].value_counts()
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie(churn_counts, labels=['Active', 'Churned'], autopct='%1.1f%%', colors=['#8fd9b6', '#ff9999'])
    ax.set_title('Customer Churn Distribution')
    return fig


model = joblib.load("churn_model.pkl")  # Make sure this file exists

# Define the prediction function
def predict_churn(input_data):
    """
    Predict churn from input data.
    input_data should be a list or array like [recency, frequency, monetary]
    """
    data_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(data_array)
    return prediction

# =========================
# Step 4: Run This File Alone
# =========================

if __name__ == "__main__":
    train_churn_model()
