# clv_model.py

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

def prepare_rfm_data(df):
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    analysis_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': [lambda x: (x.max() - x.min()).days,
                        lambda x: (analysis_date - x.min()).days],
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['recency', 'T', 'frequency', 'monetary_value']
    rfm = rfm.reset_index()
    return rfm

def fit_and_predict_clv(rfm):
    rfm = rfm[
        (rfm['frequency'] > 1) &
        (rfm['recency'] > 0) &
        (rfm['T'] > 0) &
        (rfm['monetary_value'] > 0)
    ]
    bgf = BetaGeoFitter(penalizer_coef=10.0)
    bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'])
    
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm['frequency'], rfm['monetary_value'])
    
    rfm['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        180, rfm['frequency'], rfm['recency'], rfm['T'])
    rfm['predicted_avg_value'] = ggf.conditional_expected_average_profit(
        rfm['frequency'], rfm['monetary_value'])
    rfm['predicted_clv'] = ggf.customer_lifetime_value(
        bgf, rfm['frequency'], rfm['recency'], rfm['T'], rfm['monetary_value'],
        time=6, freq='D', discount_rate=0.01)
    return rfm
